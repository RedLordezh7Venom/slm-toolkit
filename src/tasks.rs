use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use crate::types::AppMsg;

/// Search HuggingFace via the public REST API
pub async fn hf_search(query: String, tx: mpsc::UnboundedSender<AppMsg>) {
    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&limit=30",
        urlencoding::encode(&query)
    );
    match reqwest::get(&url).await {
        Err(e) => { let _ = tx.send(AppMsg::HFStatus(format!("Error: {e}"))); }
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Err(e) => { let _ = tx.send(AppMsg::HFStatus(format!("Parse error: {e}"))); }
            Ok(json) => {
                let ids: Vec<String> = json.as_array()
                    .map(|arr| arr.iter()
                        .filter_map(|v| v["id"].as_str().map(str::to_string))
                        .collect())
                    .unwrap_or_default();
                let _ = tx.send(AppMsg::HFResults(ids));
            }
        }
    }
}

/// Download a HuggingFace repo using huggingface-cli (already in venv)
pub async fn hf_download(repo: String, tx: mpsc::UnboundedSender<AppMsg>) {
    let hf_cli = "./llama.cpp/.venv/bin/huggingface-cli";
    let cli = if std::path::Path::new(hf_cli).exists() { hf_cli } else { "huggingface-cli" };

    let _ = tx.send(AppMsg::HFStatus(format!("Downloading {}…", repo)));
    let result = std::process::Command::new(cli)
        .args(["download", &repo])
        .output();
    match result {
        Ok(out) if out.status.success() => {
            let _ = tx.send(AppMsg::HFStatus(format!("✓ Downloaded {}", repo)));
        }
        Ok(out) => {
            let err = String::from_utf8_lossy(&out.stderr);
            let _ = tx.send(AppMsg::HFStatus(format!("✗ {}", err.lines().last().unwrap_or("failed"))));
        }
        Err(e) => { let _ = tx.send(AppMsg::HFStatus(format!("✗ {e}"))); }
    }
}

/// Run conversion commands sequentially, streaming stdout line by line
pub async fn run_job(cmds: Vec<(String, Vec<String>)>, tx: mpsc::UnboundedSender<AppMsg>) {
    let total = cmds.len();
    for (i, (label, argv)) in cmds.iter().enumerate() {
        let _ = tx.send(AppMsg::JobStep(i));
        let _ = tx.send(AppMsg::JobLine(format!("─── Step {}/{}: {} ───", i+1, total, label)));
        let _ = tx.send(AppMsg::JobLine(format!("$ {}", argv.join(" "))));

        let mut child = match Command::new(&argv[0])
            .args(&argv[1..])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(AppMsg::JobLine(format!("[ERROR] {e}")));
                let _ = tx.send(AppMsg::JobDone(false));
                return;
            }
        };

        // Stream stdout
        if let Some(stdout) = child.stdout.take() {
            let tx2 = tx.clone();
            let mut lines = BufReader::new(stdout).lines();
            tokio::spawn(async move {
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx2.send(AppMsg::JobLine(line));
                }
            });
        }
        // Stream stderr
        if let Some(stderr) = child.stderr.take() {
            let tx3 = tx.clone();
            let mut lines = BufReader::new(stderr).lines();
            tokio::spawn(async move {
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx3.send(AppMsg::JobLine(format!("[err] {}", line)));
                }
            });
        }

        match child.wait().await {
            Ok(status) if status.success() => {
                let _ = tx.send(AppMsg::JobLine(format!("✓ Step {} complete", i+1)));
            }
            Ok(status) => {
                let _ = tx.send(AppMsg::JobLine(format!("✗ Failed (exit {})", status)));
                let _ = tx.send(AppMsg::JobDone(false));
                return;
            }
            Err(e) => {
                let _ = tx.send(AppMsg::JobLine(format!("[ERROR] wait: {e}")));
                let _ = tx.send(AppMsg::JobDone(false));
                return;
            }
        }
    }
    let _ = tx.send(AppMsg::JobStep(total));
    let _ = tx.send(AppMsg::JobDone(true));
}
