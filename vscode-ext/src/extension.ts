import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface CetpPrediction {
    predicted_runtime_sec: number;
    confidence_interval: [number, number];
    confidence_status: 'RELIABLE' | 'EXTRAPOLATED';
    confidence_warnings: string[];
    sla_flag: 'GREEN' | 'YELLOW' | 'RED';
    sla_threshold_sec: number | null;
    top_shap_features: ShapFeature[];
    model_used: string;
    complexity_level: number;
    batch_size: number;
    num_iterations: number;
}

interface ShapFeature {
    feature: string;
    shap_value: number;
    direction: string;
}

interface CetpMeasurement {
    measured_runtime_sec: number;
    measured_peak_memory_mb: number;
    measured_avg_cpu_pct: number;
    model_used: string;
    complexity_level: number;
    batch_size: number;
    num_iterations: number;
}

interface CetpProfile {
    cpu_cores: number;
    memory_total_gb: number;
    disk_type: string;
    disk_speed_class: number;
    platform: string;
    hostname: string;
}

const KNOWN_MODELS = ['resnet18', 'resnet50', 'mobilenet', 'distilbert'] as const;

// Message shape sent from the webview (form.html's script) to the extension
// host via acquireVsCodeApi().postMessage(). slaOverride is deliberately
// accepted-but-unused: see handleWebviewPredict/handleWebviewMeasure.
interface WebviewMessage {
    command: 'predict' | 'measure' | 'newPrediction' | 'runCommand';
    model?: string;
    complexity?: string;
    cpuCores?: string;
    memoryGb?: string;
    slaOverride?: string;
    id?: string;
}

export function activate(context: vscode.ExtensionContext) {
    console.log('CETP extension activated');

    const provider = new CetpSidebarProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('cetp.predictionPanel', provider)
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('cetp.predictRuntime', () => predictRuntime(provider)),
        vscode.commands.registerCommand('cetp.measureAndCompare', () => measureAndCompare(provider)),
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

async function runCetpCommand(args: string, timeoutMs: number = 30000): Promise<string> {
    const config = vscode.workspace.getConfiguration('cetp');
    const useLocalCli = config.get<boolean>('useLocalCli', true);

    if (useLocalCli) {
        try {
            const { stdout } = await execAsync(`cetp ${args}`, { timeout: timeoutMs });
            return stdout.trim();
        } catch {
            const { stdout } = await execAsync(`python3 -m cetp ${args}`, { timeout: timeoutMs });
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

// Builds the exact `cetp predict ...` invocation used by both the Command
// Palette flow (predictRuntime, after quickpicks) and the webview form flow
// (handleWebviewPredict, after form submit) — one command builder, two entry
// points, so neither path can drift from the other.
function buildPredictCommand(model: string, complexity: number, cpuCores?: string, memoryGb?: string): string {
    const config = vscode.workspace.getConfiguration('cetp');
    const extraArgs = (cpuCores && memoryGb) ? `--cpu-cores ${cpuCores} --memory-gb ${memoryGb}` : '';
    const slaPath = config.get<string>('slaFilePath', '');
    const slaArg = slaPath ? `--sla "${slaPath}"` : '';
    return `predict --model ${model} --complexity ${complexity} ${extraArgs} ${slaArg} --json`;
}

function buildMeasureCommand(model: string, complexity: number): string {
    return `measure --model ${model} --complexity ${complexity} --json`;
}

// Runs a predict command and renders the result into the webview — shared by
// the Command Palette flow (wrapped in a notification progress) and the
// webview form flow (wrapped in an in-panel loading state).
async function runPredictionFlow(command: string, provider: CetpSidebarProvider): Promise<void> {
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
        const message = `CETP: ${prediction.predicted_runtime_sec.toFixed(2)}s predicted — SLA ${slaEmoji} ${prediction.sla_flag}`;

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

async function predictRuntime(provider: CetpSidebarProvider): Promise<void> {
    const model = await vscode.window.showQuickPick(
        ['resnet18', 'resnet50', 'mobilenet', 'distilbert'],
        {
            placeHolder: 'Select model',
            title: 'CETP: Predict Runtime — Step 1 of 3'
        }
    );
    if (!model) return;

    const complexityStr = await vscode.window.showQuickPick(
        ['1 — Minimal', '2 — Low', '3 — Medium', '4 — High', '5 — Maximum'],
        {
            placeHolder: 'Select workload complexity',
            title: 'CETP: Predict Runtime — Step 2 of 3'
        }
    );
    if (!complexityStr) return;
    const complexity = parseInt(complexityStr[0]);

    const specMode = await vscode.window.showQuickPick(
        [
            'Auto-profile this machine',
            'Specify hardware manually'
        ],
        {
            placeHolder: 'Hardware configuration',
            title: 'CETP: Predict Runtime — Step 3 of 3'
        }
    );
    if (!specMode) return;

    let cpuCoresStr: string | undefined;
    let memoryStr: string | undefined;

    if (specMode === 'Specify hardware manually') {
        cpuCoresStr = await vscode.window.showInputBox({
            prompt: 'CPU cores',
            placeHolder: 'e.g. 64',
            validateInput: (v) => isNaN(parseInt(v)) ? 'Must be a number' : null
        });
        if (!cpuCoresStr) return;

        memoryStr = await vscode.window.showInputBox({
            prompt: 'RAM in GB',
            placeHolder: 'e.g. 64',
            validateInput: (v) => isNaN(parseFloat(v)) ? 'Must be a number' : null
        });
        if (!memoryStr) return;
    }

    const command = buildPredictCommand(model, complexity, cpuCoresStr, memoryStr);

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'CETP: Running prediction...',
            cancellable: false
        },
        () => runPredictionFlow(command, provider)
    );
}

// Matches measure_calibration.py's per-complexity target range — this is a
// rough estimate of THIS process's own calibration target, not a guarantee
// for whatever machine the extension actually runs on. This is the single
// source of truth for that estimate on the TS side (there's no importable
// Python constant to pull from — measure_calibration.py only states the
// 90/105/120/180/210s range in a docstring, it derives MEASURE_NUM_ITERATIONS
// from it rather than exposing it as data) — every duration estimate shown
// in the webview (including the progress bar added below) reads from here,
// not a second copy.
const COMPLEXITY_TARGET_SECONDS: Record<number, number> = { 1: 90, 2: 105, 3: 120, 4: 180, 5: 210 };

// `cetp predict` is a near-instant CLI call (no real workload runs), so its
// progress bar just needs a short estimate to give brief visible motion —
// it will almost always finish before the bar reaches 95%.
const PREDICT_ESTIMATE_SECONDS = 3;

// Runs measure then predict-on-target and renders the comparison — shared by
// the Command Palette flow and the webview form flow, same reasoning as
// runPredictionFlow above. onProgress lets each entry point surface the
// in-flight status its own way (notification progress vs. in-panel
// progress bar); the estimateSec argument tells the webview's progress bar
// how long the *next* phase is expected to take.
async function runMeasureCompareFlow(
    measureCommand: string,
    predictCommand: string,
    provider: CetpSidebarProvider,
    onProgress: (message: string, estimateSec: number) => void
): Promise<void> {
    try {
        // measure runs a real 60-400+ second workload — needs a much longer
        // exec timeout than predict's near-instant default.
        const measureOutput = await runCetpCommand(measureCommand, 600000);

        let measurement: CetpMeasurement;
        try {
            measurement = JSON.parse(measureOutput);
        } catch {
            vscode.window.showInformationMessage(`CETP: ${measureOutput}`);
            provider.showMessage(measureOutput);
            return;
        }

        onProgress('Benchmark complete — predicting on target hardware...', PREDICT_ESTIMATE_SECONDS);

        const predictOutput = await runCetpCommand(predictCommand);

        let prediction: CetpPrediction;
        try {
            prediction = JSON.parse(predictOutput);
        } catch {
            vscode.window.showInformationMessage(`CETP: ${predictOutput}`);
            provider.showMessage(predictOutput);
            return;
        }

        provider.showComparison(measurement, prediction);

        const slaEmoji = prediction.sla_flag === 'GREEN' ? '✓' :
                          prediction.sla_flag === 'YELLOW' ? '⚠' : '✗';
        const message = `CETP: measured ${measurement.measured_runtime_sec.toFixed(2)}s here — ` +
            `predicted ${prediction.predicted_runtime_sec.toFixed(2)}s on target — SLA ${slaEmoji} ${prediction.sla_flag}`;

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
            vscode.window.showErrorMessage(`CETP measure/compare failed: ${msg}`);
        }
        provider.showError(msg);
    }
}

async function measureAndCompare(provider: CetpSidebarProvider): Promise<void> {
    const model = await vscode.window.showQuickPick(
        ['resnet18', 'resnet50', 'mobilenet', 'distilbert'],
        {
            placeHolder: 'Select model',
            title: 'CETP: Measure and Compare — Step 1 of 3'
        }
    );
    if (!model) return;

    const complexityStr = await vscode.window.showQuickPick(
        ['1 — Minimal', '2 — Low', '3 — Medium', '4 — High', '5 — Maximum'],
        {
            placeHolder: 'Select workload complexity',
            title: 'CETP: Measure and Compare — Step 2 of 3'
        }
    );
    if (!complexityStr) return;
    const complexity = parseInt(complexityStr[0]);

    const cpuCoresStr = await vscode.window.showInputBox({
        prompt: 'Target hardware CPU cores',
        placeHolder: 'e.g. 64',
        title: 'CETP: Measure and Compare — Step 3 of 3',
        validateInput: (v) => isNaN(parseInt(v)) ? 'Must be a number' : null
    });
    if (!cpuCoresStr) return;

    const memoryStr = await vscode.window.showInputBox({
        prompt: 'Target hardware RAM in GB',
        placeHolder: 'e.g. 64',
        validateInput: (v) => isNaN(parseFloat(v)) ? 'Must be a number' : null
    });
    if (!memoryStr) return;

    const measureCommand = buildMeasureCommand(model, complexity);
    const predictCommand = buildPredictCommand(model, complexity, cpuCoresStr, memoryStr);
    const estimateSec = COMPLEXITY_TARGET_SECONDS[complexity] ?? 120;

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'CETP: Measure and Compare',
            cancellable: false
        },
        async (progress) => {
            progress.report({
                message: `Running local benchmark for ${model} (complexity ${complexity})... this will take approximately ${estimateSec}s`
            });
            await runMeasureCompareFlow(measureCommand, predictCommand, provider, (message) => progress.report({ message }));
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

// Entry point for the webview form's "Predict Runtime" button — parallel to
// predictRuntime() above, but reads params straight from the form's
// postMessage instead of a chain of quickpicks, then hands off to the same
// buildPredictCommand + runPredictionFlow used by the palette command.
async function handleWebviewPredict(provider: CetpSidebarProvider, msg: WebviewMessage): Promise<void> {
    const model = msg.model ?? '';
    const complexity = parseInt(msg.complexity ?? '', 10);

    if (!KNOWN_MODELS.includes(model as typeof KNOWN_MODELS[number])) {
        provider.showError(`Invalid model: ${model}`);
        return;
    }
    if (isNaN(complexity) || complexity < 1 || complexity > 5) {
        provider.showError(`Invalid complexity level: ${msg.complexity}`);
        return;
    }
    const cpuCores = msg.cpuCores?.trim() || undefined;
    const memoryGb = msg.memoryGb?.trim() || undefined;
    if ((cpuCores && isNaN(parseInt(cpuCores, 10))) || (memoryGb && isNaN(parseFloat(memoryGb)))) {
        provider.showError('CPU cores and Memory GB must be numeric.');
        return;
    }

    const command = buildPredictCommand(model, complexity, cpuCores, memoryGb);
    provider.showLoading('Running prediction...', PREDICT_ESTIMATE_SECONDS);
    await runPredictionFlow(command, provider);
}

// Entry point for the webview form's "Measure and Compare" button — parallel
// to measureAndCompare() above, same params-from-message pattern.
async function handleWebviewMeasure(provider: CetpSidebarProvider, msg: WebviewMessage): Promise<void> {
    const model = msg.model ?? '';
    const complexity = parseInt(msg.complexity ?? '', 10);

    if (!KNOWN_MODELS.includes(model as typeof KNOWN_MODELS[number])) {
        provider.showError(`Invalid model: ${model}`);
        return;
    }
    if (isNaN(complexity) || complexity < 1 || complexity > 5) {
        provider.showError(`Invalid complexity level: ${msg.complexity}`);
        return;
    }
    const cpuCores = msg.cpuCores?.trim() || undefined;
    const memoryGb = msg.memoryGb?.trim() || undefined;
    if ((cpuCores && isNaN(parseInt(cpuCores, 10))) || (memoryGb && isNaN(parseFloat(memoryGb)))) {
        provider.showError('CPU cores and Memory GB must be numeric.');
        return;
    }

    const measureCommand = buildMeasureCommand(model, complexity);
    const predictCommand = buildPredictCommand(model, complexity, cpuCores, memoryGb);
    const estimateSec = COMPLEXITY_TARGET_SECONDS[complexity] ?? 120;

    provider.showLoading(`Running local benchmark for ${model} (complexity ${complexity})...`, estimateSec);
    await runMeasureCompareFlow(measureCommand, predictCommand, provider, (message, phaseEstimateSec) => provider.showLoading(message, phaseEstimateSec));
}

class CetpSidebarProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    resolveWebviewView(webviewView: vscode.WebviewView): void {
        this._view = webviewView;
        webviewView.webview.options = { enableScripts: true };
        webviewView.webview.html = this._getFormHtml();

        // Standard webview<->extension-host bridge: the webview runs in an
        // isolated context (no Node/vscode API access), so form submissions
        // and command-reference clicks travel here as postMessage payloads
        // and are dispatched to the same predict/measure/command logic the
        // Command Palette entry points already use.
        webviewView.webview.onDidReceiveMessage(async (message: WebviewMessage) => {
            switch (message.command) {
                case 'predict':
                    await handleWebviewPredict(this, message);
                    break;
                case 'measure':
                    await handleWebviewMeasure(this, message);
                    break;
                case 'newPrediction':
                    this.showForm();
                    break;
                case 'runCommand':
                    if (message.id) {
                        vscode.commands.executeCommand(message.id);
                    }
                    break;
            }
        });
    }

    showForm(): void {
        if (!this._view) return;
        this._view.webview.html = this._getFormHtml();
    }

    showLoading(message: string, estimateSec: number): void {
        if (!this._view) return;
        this._view.webview.html = this._getLoadingHtml(message, estimateSec);
        this._view.show(true);
    }

    showPrediction(prediction: CetpPrediction): void {
        if (!this._view) return;
        this._view.webview.html = this._getPredictionHtml(prediction);
        this._view.show(true);
    }

    showComparison(measurement: CetpMeasurement, prediction: CetpPrediction): void {
        if (!this._view) return;
        this._view.webview.html = this._getComparisonHtml(measurement, prediction);
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

    // Small reference list of every contributed CETP command, appended to the
    // form so nothing is hidden behind the Command Palette. Each entry posts
    // 'runCommand' back to the extension host, which just calls
    // vscode.commands.executeCommand — no logic lives here.
    private _getCommandsReferenceHtml(): string {
        return `
  <details class="commands">
    <summary>All CETP Commands</summary>
    <button class="command-link" data-cmd="cetp.predictRuntime">CETP: Predict Runtime</button>
    <button class="command-link" data-cmd="cetp.measureAndCompare">CETP: Measure and Compare</button>
    <button class="command-link" data-cmd="cetp.showProfile">CETP: Show Machine Profile</button>
    <button class="command-link" data-cmd="cetp.showInfo">CETP: Show Info</button>
    <button class="command-link" data-cmd="cetp.openSettings">CETP: Open Settings</button>
  </details>
  <script>
    (function() {
      const vscode = acquireVsCodeApi();
      document.querySelectorAll('.command-link').forEach(function(btn) {
        btn.addEventListener('click', function() {
          vscode.postMessage({ command: 'runCommand', id: btn.getAttribute('data-cmd') });
        });
      });
      window.__cetpVscodeApi = vscode;
    })();
  </script>`;
    }

    private _getFormStyles(): string {
        return `
  body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); font-size: 12px; }
  h2 { font-size: 14px; margin: 0 0 4px 0; }
  .subtitle { font-size: 11px; color: var(--vscode-descriptionForeground); margin-bottom: 14px; }
  label { display: block; font-size: 11px; color: var(--vscode-descriptionForeground); margin: 10px 0 4px 0; }
  select, input[type="number"] {
    width: 100%; box-sizing: border-box; padding: 4px 6px;
    background: var(--vscode-dropdown-background, var(--vscode-input-background));
    color: var(--vscode-dropdown-foreground, var(--vscode-input-foreground));
    border: 1px solid var(--vscode-dropdown-border, var(--vscode-widget-border));
    border-radius: 2px; font-family: var(--vscode-font-family); font-size: 12px;
  }
  input[type="number"] { background: var(--vscode-input-background); color: var(--vscode-input-foreground); border-color: var(--vscode-input-border, var(--vscode-widget-border)); }
  select:focus, input:focus { outline: 1px solid var(--vscode-focusBorder); outline-offset: -1px; }
  .field-hint { font-size: 10px; color: var(--vscode-descriptionForeground); margin-top: 3px; }
  .actions { display: flex; flex-direction: column; gap: 6px; margin-top: 16px; }
  button { border: none; border-radius: 2px; padding: 6px 10px; font-size: 12px; cursor: pointer; font-family: var(--vscode-font-family); }
  button.primary { background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
  button.primary:hover { background: var(--vscode-button-hoverBackground); }
  button.secondary { background: var(--vscode-button-secondaryBackground, transparent); color: var(--vscode-button-secondaryForeground, var(--vscode-foreground)); border: 1px solid var(--vscode-widget-border); }
  button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground, var(--vscode-toolbar-hoverBackground)); }
  details.commands { margin-top: 20px; border-top: 1px solid var(--vscode-widget-border); padding-top: 10px; }
  details.commands summary { cursor: pointer; font-size: 11px; font-weight: bold; color: var(--vscode-descriptionForeground); text-transform: uppercase; letter-spacing: 0.5px; }
  .command-link { display: block; width: 100%; text-align: left; background: none; border: none; color: var(--vscode-textLink-foreground); padding: 5px 0; font-size: 11px; }
  .command-link:hover { text-decoration: underline; color: var(--vscode-textLink-activeForeground); background: none; }`;
    }

    private _getFormHtml(): string {
        return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>${this._getFormStyles()}</style>
</head>
<body>
  <h2>CETP Predictions</h2>
  <div class="subtitle">Configure a workload, then predict or measure.</div>

  <label for="model">Model</label>
  <select id="model">
    <option value="resnet18">resnet18</option>
    <option value="resnet50">resnet50</option>
    <option value="mobilenet">mobilenet</option>
    <option value="distilbert">distilbert</option>
  </select>

  <label for="complexity">Complexity Level</label>
  <select id="complexity">
    <option value="1">1 — Minimal</option>
    <option value="2">2 — Low</option>
    <option value="3" selected>3 — Medium</option>
    <option value="4">4 — High</option>
    <option value="5">5 — Maximum</option>
  </select>

  <label for="cpuCores">Target CPU Cores</label>
  <input type="number" id="cpuCores" min="1" step="1" placeholder="e.g. 64">
  <div class="field-hint">Optional — leave blank to auto-profile this machine.</div>

  <label for="memoryGb">Target Memory (GB)</label>
  <input type="number" id="memoryGb" min="0" step="0.5" placeholder="e.g. 64">
  <div class="field-hint">Optional — leave blank to auto-profile this machine.</div>

  <label for="slaOverride">SLA Limit Override (sec)</label>
  <input type="number" id="slaOverride" min="0" step="1" placeholder="e.g. 30">
  <div class="field-hint">Optional. Not yet wired to the CLI — has no effect on the result.</div>

  <div class="actions">
    <button class="primary" id="predictBtn">Predict Runtime</button>
    <button class="primary" id="measureBtn">Measure and Compare</button>
  </div>

  ${this._getCommandsReferenceHtml()}

  <script>
    (function() {
      const vscode = window.__cetpVscodeApi || acquireVsCodeApi();

      function readForm(command) {
        return {
          command: command,
          model: document.getElementById('model').value,
          complexity: document.getElementById('complexity').value,
          cpuCores: document.getElementById('cpuCores').value,
          memoryGb: document.getElementById('memoryGb').value,
          slaOverride: document.getElementById('slaOverride').value
        };
      }

      document.getElementById('predictBtn').addEventListener('click', function() {
        vscode.postMessage(readForm('predict'));
      });
      document.getElementById('measureBtn').addEventListener('click', function() {
        vscode.postMessage(readForm('measure'));
      });
    })();
  </script>
</body>
</html>`;
    }

    // estimateSec drives a TIME-BASED ESTIMATE, not real progress: `cetp
    // measure`/`cetp predict` are blocking subprocess calls with no
    // intermediate progress signal to observe. The bar and rotating status
    // phrases below are purely elapsed-time-vs-known-target cosmetics —
    // they're capped at 95% precisely so a slow real run is never shown as
    // "done" before the actual result HTML replaces this page outright (see
    // runPredictionFlow/runMeasureCompareFlow, which call provider.showX()
    // the moment the real CLI output is parsed, independent of the bar's
    // state). If the real call finishes early, this whole document is
    // discarded mid-animation when webview.html is reassigned — there is no
    // "wait for the bar to catch up" step.
    private _getLoadingHtml(message: string, estimateSec: number): string {
        return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 24px 16px; color: var(--vscode-foreground); font-size: 12px; }
  .loading-heading { color: var(--vscode-foreground); line-height: 1.5; margin-bottom: 16px; text-align: center; }
  .progress-track {
    width: 100%; height: 6px; border-radius: 3px; overflow: hidden;
    background: var(--vscode-widget-border);
  }
  .progress-fill {
    height: 100%; width: 0%; border-radius: 3px;
    background: var(--vscode-progressBar-background, var(--vscode-focusBorder));
    transition: width 0.3s linear;
  }
  .elapsed-text { margin-top: 8px; font-size: 11px; color: var(--vscode-descriptionForeground); text-align: center; }
  .status-text { margin-top: 12px; font-size: 11px; font-style: italic; color: var(--vscode-descriptionForeground); text-align: center; min-height: 14px; }
</style>
</head>
<body>
  <div class="loading-heading">${message}</div>
  <div class="progress-track"><div class="progress-fill" id="cetpProgressFill"></div></div>
  <div class="elapsed-text" id="cetpElapsedText">Running... 0s elapsed (est. ${estimateSec}s)</div>
  <div class="status-text" id="cetpStatusText"></div>

  <script>
    (function() {
      // Cosmetic pacing only — see the comment on _getLoadingHtml in
      // extension.ts. Nothing here reads real subprocess state; it is
      // elapsed wall-clock time compared against the known calibration
      // estimate passed in from the extension host.
      var estimateSec = ${estimateSec};
      var startTime = Date.now();
      var fill = document.getElementById('cetpProgressFill');
      var elapsedEl = document.getElementById('cetpElapsedText');
      var statusEl = document.getElementById('cetpStatusText');

      // Illustrative-only phrasing to make the wait feel active. These do
      // NOT correspond to real stages of the underlying CLI call — cetp
      // measure/predict give no intermediate stage signal to surface.
      var statusPhrases = [
        'Loading model weights...',
        'Running inference batches...',
        'Measuring memory usage...',
        'Collecting timing samples...'
      ];

      function tick() {
        var elapsedSec = (Date.now() - startTime) / 1000;
        // Capped at 95% — real completion swaps in the result page directly,
        // it never comes from this bar reaching 100%.
        var pct = Math.min(95, (elapsedSec / estimateSec) * 95);
        fill.style.width = pct + '%';
        elapsedEl.textContent = 'Running... ' + Math.floor(elapsedSec) + 's elapsed (est. ' + estimateSec + 's)';
        var phraseIndex = Math.floor(elapsedSec / 15) % statusPhrases.length;
        statusEl.textContent = statusPhrases[phraseIndex];
      }

      tick();
      setInterval(tick, 250);
    })();
  </script>
</body>
</html>`;
    }

    private _getNewPredictionButtonHtml(): string {
        return `
  <div class="actions" style="margin-top:16px;">
    <button class="secondary" id="newPredictionBtn">New Prediction</button>
  </div>
  <script>
    (function() {
      const vscode = window.__cetpVscodeApi || acquireVsCodeApi();
      document.getElementById('newPredictionBtn').addEventListener('click', function() {
        vscode.postMessage({ command: 'newPrediction' });
      });
    })();
  </script>`;
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

        const extrapolationWarning = p.confidence_status === 'EXTRAPOLATED' ? `
  <div class="extrapolation-warning">
    <div class="extrapolation-title">⚠ Outside Validated Hardware Range</div>
    <ul>
      ${p.confidence_warnings.map(w => `<li>${w}</li>`).join('\n      ')}
    </ul>
    <div class="extrapolation-note">Predictions on hardware outside the training range may be significantly less accurate.</div>
  </div>
` : '';

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
  .extrapolation-warning { margin-bottom: 14px; padding: 8px 10px; background: rgba(255, 152, 0, 0.15); border: 1px solid #ff9800; border-left: 4px solid #ff9800; border-radius: 3px; }
  .extrapolation-title { font-weight: bold; color: #ff9800; font-size: 12px; margin-bottom: 4px; }
  .extrapolation-warning ul { margin: 4px 0; padding-left: 18px; font-size: 11px; }
  .extrapolation-warning li { margin-bottom: 2px; }
  .extrapolation-note { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 4px; }
  .actions { display: flex; flex-direction: column; gap: 6px; }
  button { border: none; border-radius: 2px; padding: 6px 10px; font-size: 12px; cursor: pointer; font-family: var(--vscode-font-family); }
  button.secondary { background: var(--vscode-button-secondaryBackground, transparent); color: var(--vscode-button-secondaryForeground, var(--vscode-foreground)); border: 1px solid var(--vscode-widget-border); }
  button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground, var(--vscode-toolbar-hoverBackground)); }
</style>
</head>
<body>
  ${extrapolationWarning}
  <h2>${p.model_used} / complexity ${p.complexity_level}</h2>

  <div class="metric">
    <div class="label">Predicted runtime</div>
    <div class="value">${p.predicted_runtime_sec.toFixed(2)}s</div>
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

  ${this._getNewPredictionButtonHtml()}
</body>
</html>`;
    }

    private _getComparisonHtml(m: CetpMeasurement, p: CetpPrediction): string {
        // Deliberately not reusing _getPredictionHtml — duplicated here so the
        // standalone predict command's render path is never touched by this change.
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

        const extrapolationWarning = p.confidence_status === 'EXTRAPOLATED' ? `
  <div class="extrapolation-warning">
    <div class="extrapolation-title">⚠ Outside Validated Hardware Range</div>
    <ul>
      ${p.confidence_warnings.map(w => `<li>${w}</li>`).join('\n      ')}
    </ul>
    <div class="extrapolation-note">Predictions on hardware outside the training range may be significantly less accurate.</div>
  </div>
` : '';

        // batch_size is shared by design (see cli.py's measure command — it takes
        // batch_size from the same get_workload_params() predict uses); num_iterations
        // is deliberately NOT shared (measure's is locally recalibrated timing, not a
        // predictor feature — see measure_calibration.py). Surfacing both, not just
        // one, so neither number is silently assumed comparable to the other.
        const batchSizeMismatch = m.batch_size !== p.batch_size;
        const batchBanner = batchSizeMismatch
            ? `<div class="mismatch-warning">⚠ Batch size differs between measurement (${m.batch_size}) and prediction (${p.batch_size}) — these two runs do not describe comparable work.</div>`
            : `<div class="shared-batch">Batch size: <strong>${m.batch_size}</strong> — shared by both runs below; both numbers describe the same unit of work.</div>`;

        return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); font-size: 12px; }
  h2 { font-size: 13px; margin: 0 0 4px 0; }
  .subtitle { font-size: 11px; color: var(--vscode-descriptionForeground); margin-bottom: 12px; }
  .shared-batch { margin-bottom: 12px; padding: 6px 10px; background: var(--vscode-textBlockQuote-background); border-left: 3px solid var(--vscode-textBlockQuote-border); font-size: 11px; }
  .mismatch-warning { margin-bottom: 12px; padding: 6px 10px; background: rgba(244, 67, 54, 0.15); border-left: 3px solid #f44336; font-size: 11px; }
  .sla-hero { text-align: center; padding: 10px; margin-bottom: 14px; border-radius: 4px; background: ${slaColor}; color: white; }
  .sla-hero .flag { font-size: 18px; font-weight: bold; }
  .sla-hero .limit { font-size: 11px; opacity: 0.9; margin-top: 2px; }
  .columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 14px; }
  .column { padding: 8px; border: 1px solid var(--vscode-widget-border); border-radius: 4px; }
  .column-title { font-size: 10px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; color: var(--vscode-descriptionForeground); margin-bottom: 6px; }
  .metric { margin-bottom: 8px; }
  .label { color: var(--vscode-descriptionForeground); font-size: 11px; }
  .value { font-size: 15px; font-weight: bold; }
  .confidence { font-size: 10px; color: var(--vscode-descriptionForeground); }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 11px; }
  th { text-align: left; color: var(--vscode-descriptionForeground); padding: 3px 0; border-bottom: 1px solid var(--vscode-widget-border); }
  td { padding: 3px 0; }
  .section { margin-top: 14px; }
  .section-title { font-size: 11px; font-weight: bold; color: var(--vscode-descriptionForeground); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .model-badge { font-size: 10px; color: var(--vscode-descriptionForeground); margin-top: 4px; }
  .extrapolation-warning { margin-bottom: 14px; padding: 8px 10px; background: rgba(255, 152, 0, 0.15); border: 1px solid #ff9800; border-left: 4px solid #ff9800; border-radius: 3px; }
  .extrapolation-title { font-weight: bold; color: #ff9800; font-size: 12px; margin-bottom: 4px; }
  .extrapolation-warning ul { margin: 4px 0; padding-left: 18px; font-size: 11px; }
  .extrapolation-warning li { margin-bottom: 2px; }
  .extrapolation-note { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 4px; }
  .actions { display: flex; flex-direction: column; gap: 6px; }
  button { border: none; border-radius: 2px; padding: 6px 10px; font-size: 12px; cursor: pointer; font-family: var(--vscode-font-family); }
  button.secondary { background: var(--vscode-button-secondaryBackground, transparent); color: var(--vscode-button-secondaryForeground, var(--vscode-foreground)); border: 1px solid var(--vscode-widget-border); }
  button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground, var(--vscode-toolbar-hoverBackground)); }
</style>
</head>
<body>
  <h2>${p.model_used} — Measured vs. Predicted</h2>
  <div class="subtitle">complexity ${p.complexity_level}</div>

  ${batchBanner}

  <div class="sla-hero">
    <div class="flag">${slaIcon} SLA: ${p.sla_flag}</div>
    <div class="limit">limit: ${p.sla_threshold_sec}s</div>
  </div>

  ${extrapolationWarning}

  <div class="columns">
    <div class="column">
      <div class="column-title">Measured on this machine</div>
      <div class="metric">
        <div class="label">Runtime</div>
        <div class="value">${m.measured_runtime_sec.toFixed(2)}s</div>
      </div>
      <div class="metric">
        <div class="label">Peak memory</div>
        <div class="value">${m.measured_peak_memory_mb.toFixed(0)}MB</div>
      </div>
      <div class="metric">
        <div class="label">Avg CPU</div>
        <div class="value">${m.measured_avg_cpu_pct.toFixed(0)}%</div>
      </div>
    </div>

    <div class="column">
      <div class="column-title">Predicted on target hardware</div>
      <div class="metric">
        <div class="label">Runtime</div>
        <div class="value">${p.predicted_runtime_sec.toFixed(2)}s</div>
        <div class="confidence">${p.confidence_interval[0].toFixed(2)}s — ${p.confidence_interval[1].toFixed(2)}s (90% CI)</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Top factors (predicted side)</div>
    <table>
      <tr><th>Feature</th><th>Impact</th></tr>
      ${shapRows}
    </table>
  </div>

  <div class="model-badge">Measured iterations: ${m.num_iterations} (local timing calibration) · Predicted iterations: ${p.num_iterations} (trained model feature, not comparable to the measured count)</div>

  ${this._getNewPredictionButtonHtml()}
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
  .actions { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
  button { border: none; border-radius: 2px; padding: 6px 10px; font-size: 12px; cursor: pointer; font-family: var(--vscode-font-family); }
  button.secondary { background: var(--vscode-button-secondaryBackground, transparent); color: var(--vscode-button-secondaryForeground, var(--vscode-foreground)); border: 1px solid var(--vscode-widget-border); }
  button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground, var(--vscode-toolbar-hoverBackground)); }
</style>
</head>
<body>
  <pre>${message}</pre>

  ${this._getNewPredictionButtonHtml()}
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
  .actions { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
  button { border: none; border-radius: 2px; padding: 6px 10px; font-size: 12px; cursor: pointer; font-family: var(--vscode-font-family); }
  button.secondary { background: var(--vscode-button-secondaryBackground, transparent); color: var(--vscode-button-secondaryForeground, var(--vscode-foreground)); border: 1px solid var(--vscode-widget-border); }
  button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground, var(--vscode-toolbar-hoverBackground)); }
</style>
</head>
<body>
  <div class="error">✗ Prediction failed</div>
  <p>${error}</p>
  <p>Make sure cetp is installed: <code>pip install cetp</code></p>

  ${this._getNewPredictionButtonHtml()}
</body>
</html>`;
    }
}

