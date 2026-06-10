import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface CetpPrediction {
    runtime_sec: number;
    confidence_interval: [number, number];
    sla_flag: 'GREEN' | 'YELLOW' | 'RED';
    sla_threshold_sec: number;
    top_shap_features: ShapFeature[];
    model_used: string;
    workload_type: string;
    workload_name: string;
}

interface ShapFeature {
    feature: string;
    shap_value: number;
    direction: string;
}

interface CetpProfile {
    cpu_cores: number;
    memory_total_gb: number;
    disk_type: string;
    disk_speed_class: number;
    platform: string;
    hostname: string;
}

interface WorkloadInput {
    workloadType: string;
    workloadName: string;
    complexity: number;
    cpuCores?: number;
    memoryGb?: number;
    diskType?: string;
    cpuPct?: number;
    memUsedGb?: number;
}

export function activate(context: vscode.ExtensionContext) {
    console.log('CETP extension activated');

    const provider = new CetpSidebarProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('cetp.predictionPanel', provider)
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('cetp.predictRuntime', () => predictRuntime(provider)),
        vscode.commands.registerCommand('cetp.showProfile', () => showProfile()),
        vscode.commands.registerCommand('cetp.showInfo', () => showInfo()),
        vscode.commands.registerCommand('cetp.openSettings', () =>
            vscode.commands.executeCommand('workbench.action.openSettings', 'cetp')
        )
    );

    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.command = 'cetp.showInfo';
    updateStatusBar(statusBar);
    statusBar.show();
    context.subscriptions.push(statusBar);

    const interval = setInterval(() => updateStatusBar(statusBar), 30000);
    context.subscriptions.push({ dispose: () => clearInterval(interval) });
}

export function deactivate() {}

async function updateStatusBar(statusBar: vscode.StatusBarItem): Promise<void> {
    try {
        await runCetpCommand('info --json 2>/dev/null || echo "{}"');
        statusBar.text = '$(pulse) CETP';
        statusBar.tooltip = 'CETP Runtime Prediction — click for info';
    } catch {
        statusBar.text = '$(pulse) CETP';
        statusBar.tooltip = 'CETP Runtime Prediction';
    }
}

async function runCetpCommand(args: string): Promise<string> {
    const config = vscode.workspace.getConfiguration('cetp');
    const useLocalCli = config.get<boolean>('useLocalCli', true);

    if (useLocalCli) {
        try {
            const { stdout } = await execAsync(`cetp ${args}`, { timeout: 30000 });
            return stdout.trim();
        } catch {
            const { stdout } = await execAsync(`python3 -m cetp ${args}`, { timeout: 30000 });
            return stdout.trim();
        }
    } else {
        const endpoint = config.get<string>('apiEndpoint', 'http://localhost:8000');
        const { stdout } = await execAsync(
            `curl -sf ${endpoint}/health`,
            { timeout: 10000 }
        );
        return stdout.trim();
    }
}

async function predictRuntime(provider: CetpSidebarProvider): Promise<void> {
    const config = vscode.workspace.getConfiguration('cetp');

    const workloadType = await vscode.window.showQuickPick(
        ['ML', 'DB', 'WEB'],
        {
            placeHolder: 'Select workload type',
            title: 'CETP: Predict Runtime — Step 1 of 3'
        }
    );
    if (!workloadType) return;

    const workloadNames: Record<string, string[]> = {
        'ML': ['ml_resnet', 'ml_bert'],
        'DB': ['tpch_q3', 'tpch_q6'],
        'WEB': ['wrk_low', 'wrk_high'],
    };

    const workloadName = await vscode.window.showQuickPick(
        workloadNames[workloadType],
        {
            placeHolder: 'Select workload name',
            title: 'CETP: Predict Runtime — Step 2 of 3'
        }
    );
    if (!workloadName) return;

    const complexityStr = await vscode.window.showQuickPick(
        ['1 — Minimal', '2 — Low', '3 — Medium', '4 — High', '5 — Maximum'],
        {
            placeHolder: 'Select workload complexity',
            title: 'CETP: Predict Runtime — Step 3 of 3'
        }
    );
    if (!complexityStr) return;
    const complexity = parseInt(complexityStr[0]);

    const specMode = await vscode.window.showQuickPick(
        [
            'Auto-profile this machine',
            'Specify production server specs manually'
        ],
        {
            placeHolder: 'Hardware configuration',
            title: 'CETP: Hardware Source'
        }
    );
    if (!specMode) return;

    let extraArgs = '';

    if (specMode === 'Specify production server specs manually') {
        const cpuCoresStr = await vscode.window.showInputBox({
            prompt: 'Production server CPU cores',
            placeHolder: 'e.g. 64',
            validateInput: (v) => isNaN(parseInt(v)) ? 'Must be a number' : null
        });
        if (!cpuCoresStr) return;

        const memoryStr = await vscode.window.showInputBox({
            prompt: 'Production server RAM in GB',
            placeHolder: 'e.g. 256',
            validateInput: (v) => isNaN(parseFloat(v)) ? 'Must be a number' : null
        });
        if (!memoryStr) return;

        const diskType = await vscode.window.showQuickPick(
            ['SSD', 'NVMe', 'HDD'],
            { placeHolder: 'Production server disk type' }
        );
        if (!diskType) return;

        extraArgs = `--cpu-cores ${cpuCoresStr} --memory-gb ${memoryStr} --disk-type ${diskType}`;
    }

    const slaPath = config.get<string>('slaFilePath', '');
    const slaArg = slaPath ? `--sla "${slaPath}"` : '';
    const command = `predict --workload-type ${workloadType} --workload-name ${workloadName} --complexity ${complexity} ${extraArgs} ${slaArg} --json`;

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'CETP: Running prediction...',
            cancellable: false
        },
        async () => {
            try {
                const output = await runCetpCommand(command);

                let prediction: CetpPrediction;
                try {
                    prediction = JSON.parse(output);
                } catch {
                    vscode.window.showInformationMessage(`CETP: ${output}`);
                    provider.showMessage(output);
                    return;
                }

                provider.showPrediction(prediction);

                const slaEmoji = prediction.sla_flag === 'GREEN' ? '✓' :
                                  prediction.sla_flag === 'YELLOW' ? '⚠' : '✗';
                const message = `CETP: ${prediction.runtime_sec.toFixed(2)}s predicted — SLA ${slaEmoji} ${prediction.sla_flag}`;

                if (prediction.sla_flag === 'RED') {
                    vscode.window.showWarningMessage(message);
                } else {
                    vscode.window.showInformationMessage(message);
                }

            } catch (error: unknown) {
                const msg = error instanceof Error ? error.message : String(error);
                if (msg.includes('command not found') || msg.includes('No such file')) {
                    vscode.window.showErrorMessage(
                        'CETP CLI not found. Install with: pip install cetp',
                        'Open Terminal'
                    ).then(action => {
                        if (action === 'Open Terminal') {
                            vscode.commands.executeCommand('workbench.action.terminal.new');
                        }
                    });
                } else {
                    vscode.window.showErrorMessage(`CETP prediction failed: ${msg}`);
                }
                provider.showError(msg);
            }
        }
    );
}

async function showProfile(): Promise<void> {
    try {
        const output = await runCetpCommand('profile');
        vscode.window.showInformationMessage('CETP Profile', { modal: true, detail: output });
    } catch (error: unknown) {
        const msg = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`CETP: Could not get profile — ${msg}`);
    }
}

async function showInfo(): Promise<void> {
    try {
        const output = await runCetpCommand('info');
        vscode.window.showInformationMessage('CETP Info', { modal: true, detail: output });
    } catch (error: unknown) {
        const msg = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`CETP: Could not get info — ${msg}`);
    }
}

class CetpSidebarProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    resolveWebviewView(webviewView: vscode.WebviewView): void {
        this._view = webviewView;
        webviewView.webview.options = { enableScripts: true };
        webviewView.webview.html = this._getWelcomeHtml();
    }

    showPrediction(prediction: CetpPrediction): void {
        if (!this._view) return;
        this._view.webview.html = this._getPredictionHtml(prediction);
        this._view.show(true);
    }

    showMessage(message: string): void {
        if (!this._view) return;
        this._view.webview.html = this._getMessageHtml(message);
        this._view.show(true);
    }

    showError(error: string): void {
        if (!this._view) return;
        this._view.webview.html = this._getErrorHtml(error);
        this._view.show(true);
    }

    private _getWelcomeHtml(): string {
        return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); }
  h2 { font-size: 14px; margin-bottom: 8px; }
  p { font-size: 12px; color: var(--vscode-descriptionForeground); line-height: 1.5; }
  .hint { font-size: 11px; margin-top: 16px; padding: 8px; background: var(--vscode-textBlockQuote-background); border-left: 3px solid var(--vscode-textBlockQuote-border); }
</style>
</head>
<body>
  <h2>CETP Predictions</h2>
  <p>Right-click any .py, .java, or .cpp file and select <strong>CETP: Predict Runtime</strong> to get started.</p>
  <div class="hint">
    Predictions will appear here after you run a workload analysis.
  </div>
</body>
</html>`;
    }

    private _getPredictionHtml(p: CetpPrediction): string {
        const slaColor = p.sla_flag === 'GREEN' ? '#4caf50' :
                          p.sla_flag === 'YELLOW' ? '#ff9800' : '#f44336';
        const slaIcon = p.sla_flag === 'GREEN' ? '✓' :
                         p.sla_flag === 'YELLOW' ? '⚠' : '✗';

        const shapRows = p.top_shap_features.map(f => {
            const dir = f.direction === 'increases_runtime' ? '▲' : '▼';
            const color = f.direction === 'increases_runtime' ? '#f44336' : '#4caf50';
            return `<tr>
                <td>${f.feature}</td>
                <td style="color:${color}">${dir} ${Math.abs(f.shap_value).toFixed(3)}</td>
            </tr>`;
        }).join('');

        return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); font-size: 12px; }
  h2 { font-size: 13px; margin: 0 0 12px 0; }
  .metric { margin-bottom: 8px; }
  .label { color: var(--vscode-descriptionForeground); font-size: 11px; }
  .value { font-size: 16px; font-weight: bold; }
  .sla-badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-weight: bold; color: white; background: ${slaColor}; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 11px; }
  th { text-align: left; color: var(--vscode-descriptionForeground); padding: 3px 0; border-bottom: 1px solid var(--vscode-widget-border); }
  td { padding: 3px 0; }
  .section { margin-top: 14px; }
  .section-title { font-size: 11px; font-weight: bold; color: var(--vscode-descriptionForeground); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .confidence { font-size: 11px; color: var(--vscode-descriptionForeground); }
  .model-badge { font-size: 10px; color: var(--vscode-descriptionForeground); margin-top: 4px; }
</style>
</head>
<body>
  <h2>${p.workload_type} / ${p.workload_name}</h2>

  <div class="metric">
    <div class="label">Predicted runtime</div>
    <div class="value">${p.runtime_sec.toFixed(2)}s</div>
    <div class="confidence">${p.confidence_interval[0].toFixed(2)}s — ${p.confidence_interval[1].toFixed(2)}s (90% CI)</div>
  </div>

  <div class="metric">
    <div class="label">SLA status (limit: ${p.sla_threshold_sec}s)</div>
    <div><span class="sla-badge">${slaIcon} ${p.sla_flag}</span></div>
  </div>

  <div class="section">
    <div class="section-title">Top factors</div>
    <table>
      <tr><th>Feature</th><th>Impact</th></tr>
      ${shapRows}
    </table>
  </div>

  <div class="model-badge">Model: ${p.model_used}</div>
</body>
</html>`;
    }

    private _getMessageHtml(message: string): string {
        return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); font-size: 12px; }
  pre { white-space: pre-wrap; word-break: break-word; font-family: var(--vscode-editor-font-family); font-size: 11px; }
</style>
</head>
<body>
  <pre>${message}</pre>
</body>
</html>`;
    }

    private _getErrorHtml(error: string): string {
        return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); font-size: 12px; }
  .error { color: #f44336; }
  p { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 8px; }
</style>
</head>
<body>
  <div class="error">✗ Prediction failed</div>
  <p>${error}</p>
  <p>Make sure cetp is installed: <code>pip install cetp</code></p>
</body>
</html>`;
    }
}

