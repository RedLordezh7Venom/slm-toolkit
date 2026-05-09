use std::path::PathBuf;
use crate::types::ConvJob;

// ─── Conversion graph ─────────────────────────────────────────────────────────

pub fn dst_fmts_for(src: &str) -> &'static [(&'static str, &'static str)] {
    match src {
        "hf"   => &[("gguf","GGUF"), ("quantized","Quantized GGUF"), ("onnx","ONNX")],
        "gguf" => &[("quantized","Quantized GGUF")],
        "ggml" => &[("gguf","GGUF"), ("quantized","Quantized GGUF")],
        "lora" => &[("gguf","GGUF"), ("quantized","Quantized GGUF")],
        _      => &[],
    }
}

pub fn resolve_steps(src: &str, dst: &str) -> Option<Vec<&'static str>> {
    Some(match (src, dst) {
        ("hf",   "gguf")      => vec!["hf_to_gguf"],
        ("hf",   "quantized") => vec!["hf_to_gguf", "gguf_to_quant"],
        ("hf",   "onnx")      => vec!["hf_to_onnx"],
        ("gguf", "quantized") => vec!["gguf_to_quant"],
        ("ggml", "gguf")      => vec!["ggml_to_gguf"],
        ("ggml", "quantized") => vec!["ggml_to_gguf", "gguf_to_quant"],
        ("lora", "gguf")      => vec!["lora_to_gguf"],
        ("lora", "quantized") => vec!["lora_to_gguf", "gguf_to_quant"],
        _                     => return None,
    })
}

pub fn step_label(step: &str) -> &'static str {
    match step {
        "hf_to_gguf"    => "Convert HuggingFace → GGUF",
        "gguf_to_quant" => "Quantize GGUF",
        "ggml_to_gguf"  => "Convert GGML → GGUF",
        "lora_to_gguf"  => "Merge LoRA → GGUF",
        "hf_to_onnx"    => "Export HuggingFace → ONNX",
        _               => "Unknown step",
    }
}

// ─── Build concrete shell commands from a ConvJob ─────────────────────────────

pub fn build_commands(job: &ConvJob) -> Vec<(String, Vec<String>)> {
    let llama = PathBuf::from("./llama.cpp");
    let venv_py = llama.join(".venv/bin/python3");
    let py = if venv_py.exists() {
        venv_py.to_string_lossy().to_string()
    } else {
        "python3".to_string()
    };

    let stem = PathBuf::from(&job.src_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());
    let inter = format!("/tmp/{}_converted.gguf", stem);

    let outtype = if job.outtype.is_empty() { "f16" } else { &job.outtype };
    let qtype   = if job.qtype.is_empty()   { "Q4_K_M" } else { &job.qtype };

    let steps = match resolve_steps(&job.src_fmt, &job.dst_fmt) {
        Some(s) => s,
        None    => return vec![],
    };

    let mut cmds = vec![];

    for step in &steps {
        match *step {
            "hf_to_gguf" => {
                let out = if job.out_path.is_empty() { inter.clone() } else { job.out_path.clone() };
                cmds.push((step_label(step).to_string(), vec![
                    py.clone(),
                    llama.join("convert_hf_to_gguf.py").to_string_lossy().to_string(),
                    job.src_path.clone(),
                    "--outtype".into(), outtype.to_string(),
                    "--outfile".into(), out,
                ]));
            }
            "gguf_to_quant" => {
                let inp = if job.src_fmt == "gguf" { job.src_path.clone() } else { inter.clone() };
                let out = if job.out_path.is_empty() {
                    format!("{}-{}.gguf", stem, qtype)
                } else { job.out_path.clone() };
                cmds.push((step_label(step).to_string(), vec![
                    llama.join("llama-quantize").to_string_lossy().to_string(),
                    inp, out, qtype.to_string(),
                ]));
            }
            "ggml_to_gguf" => {
                cmds.push((step_label(step).to_string(), vec![
                    py.clone(),
                    llama.join("convert_llama_ggml_to_gguf.py").to_string_lossy().to_string(),
                    "--input".into(), job.src_path.clone(),
                    "--output".into(), inter.clone(),
                ]));
            }
            "lora_to_gguf" => {
                let mut args = vec![
                    py.clone(),
                    llama.join("convert_lora_to_gguf.py").to_string_lossy().to_string(),
                    job.src_path.clone(),
                ];
                if !job.base_model.is_empty() {
                    args.extend_from_slice(&["--base".into(), job.base_model.clone()]);
                }
                args.extend_from_slice(&["--outfile".into(), inter.clone()]);
                cmds.push((step_label(step).to_string(), args));
            }
            "hf_to_onnx" => {
                let out = if job.out_path.is_empty() { format!("{}_onnx", stem) } else { job.out_path.clone() };
                cmds.push((step_label(step).to_string(), vec![
                    py.clone(), "-m".into(), "optimum.exporters.onnx".into(),
                    "--model".into(), job.src_path.clone(), out,
                ]));
            }
            _ => {}
        }
    }
    cmds
}
