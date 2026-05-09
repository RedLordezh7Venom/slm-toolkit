use ratatui::widgets::ListState;
use tui_input::Input;

// ─── Constants ────────────────────────────────────────────────────────────────

pub const QUANT_TYPES: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0",
    "Q6_K",   "Q3_K_M", "Q2_K",   "IQ4_NL", "F16", "BF16",
];
pub const OUT_TYPES: &[&str] = &["f16", "bf16", "f32", "q8_0", "auto"];
pub const SRC_FMTS: &[(&str, &str)] = &[
    ("hf",   "HuggingFace Model (directory / repo ID)"),
    ("gguf", "GGUF File"),
    ("ggml", "GGML File (legacy)"),
    ("lora", "LoRA Adapter"),
];

pub const MENU_ITEMS: &[&str] = &[
    "  Convert / Quantize a Model",
    "  Browse & Download HuggingFace Models",
    "  Quit",
];

// ─── Messages from async tasks ────────────────────────────────────────────────

pub enum AppMsg {
    HFResults(Vec<String>),
    HFStatus(String),
    JobLine(String),
    JobStep(usize),
    JobDone(bool),
}

// ─── Conversion job ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ConvJob {
    pub src_fmt:    String,
    pub dst_fmt:    String,
    pub src_path:   String,
    pub out_path:   String,
    pub outtype:    String,
    pub qtype:      String,
    pub base_model: String,
}

// ─── Wizard ───────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
pub enum WizardStep {
    SrcFmt, SrcPath, DstFmt, OutType, QType, OutPath, BaseModel, Confirm,
}

pub struct WizardState {
    pub step:         WizardStep,
    pub src_fmt_idx:  usize,
    pub dst_fmt_idx:  usize,
    pub outtype_idx:  usize,
    pub qtype_idx:    usize,
    pub src_path:     Input,
    pub out_path:     Input,
    pub base_model:   Input,
    pub plan:         Vec<String>,
}

impl WizardState {
    pub fn new() -> Self {
        Self {
            step: WizardStep::SrcFmt,
            src_fmt_idx: 0, dst_fmt_idx: 0,
            outtype_idx: 0, qtype_idx: 0,
            src_path: Input::default(),
            out_path: Input::default(),
            base_model: Input::default(),
            plan: vec![],
        }
    }
    pub fn src_fmt(&self) -> &str { SRC_FMTS[self.src_fmt_idx].0 }
    pub fn dst_fmt(&self) -> &str {
        let d = crate::graph::dst_fmts_for(self.src_fmt());
        if d.is_empty() { "" } else { d[self.dst_fmt_idx.min(d.len()-1)].0 }
    }
    pub fn outtype(&self) -> &str { OUT_TYPES[self.outtype_idx] }
    pub fn qtype(&self)   -> &str { QUANT_TYPES[self.qtype_idx] }

    pub fn next_step(&self) -> WizardStep {
        match self.step {
            WizardStep::SrcFmt  => WizardStep::SrcPath,
            WizardStep::SrcPath => WizardStep::DstFmt,
            WizardStep::DstFmt  => WizardStep::OutType,
            WizardStep::OutType => if self.dst_fmt()=="quantized" { WizardStep::QType } else { WizardStep::OutPath },
            WizardStep::QType   => WizardStep::OutPath,
            WizardStep::OutPath => if self.src_fmt()=="lora" { WizardStep::BaseModel } else { WizardStep::Confirm },
            WizardStep::BaseModel => WizardStep::Confirm,
            WizardStep::Confirm => WizardStep::Confirm,
        }
    }
    pub fn prev_step(&self) -> WizardStep {
        match self.step {
            WizardStep::SrcFmt    => WizardStep::SrcFmt,
            WizardStep::SrcPath   => WizardStep::SrcFmt,
            WizardStep::DstFmt    => WizardStep::SrcPath,
            WizardStep::OutType   => WizardStep::DstFmt,
            WizardStep::QType     => WizardStep::OutType,
            WizardStep::OutPath   => if self.dst_fmt()=="quantized" { WizardStep::QType } else { WizardStep::OutType },
            WizardStep::BaseModel => WizardStep::OutPath,
            WizardStep::Confirm   => if self.src_fmt()=="lora" { WizardStep::BaseModel } else { WizardStep::OutPath },
        }
    }
    pub fn to_job(&self) -> ConvJob {
        ConvJob {
            src_fmt:    self.src_fmt().to_string(),
            dst_fmt:    self.dst_fmt().to_string(),
            src_path:   self.src_path.value().to_string(),
            out_path:   self.out_path.value().to_string(),
            outtype:    self.outtype().to_string(),
            qtype:      self.qtype().to_string(),
            base_model: self.base_model.value().to_string(),
        }
    }
    pub fn update_plan(&mut self) {
        let steps = crate::graph::resolve_steps(self.src_fmt(), self.dst_fmt());
        self.plan = match steps {
            Some(s) => s.iter().map(|s| format!("  → {}", crate::graph::step_label(s))).collect(),
            None    => vec!["  ✗ No conversion path".to_string()],
        };
    }
}

// ─── HF Search ────────────────────────────────────────────────────────────────

#[derive(PartialEq)]
pub enum HFFocus { Query, Results }

pub struct HFSearchState {
    pub query:      Input,
    pub results:    Vec<String>,
    pub list_state: ListState,
    pub status:     String,
    pub focus:      HFFocus,
}

impl HFSearchState {
    pub fn new() -> Self {
        Self {
            query: Input::default(),
            results: vec![],
            list_state: ListState::default(),
            status: "Type query, press Enter to search  |  Tab to switch focus  |  Esc back".to_string(),
            focus: HFFocus::Query,
        }
    }
}

// ─── Job Runner ───────────────────────────────────────────────────────────────

pub struct JobState {
    pub cmds:         Vec<(String, Vec<String>)>,
    pub current_step: usize,
    pub output:       Vec<String>,
    pub done:         bool,
    pub success:      bool,
    pub progress:     u16,
}

impl JobState {
    pub fn new(cmds: Vec<(String, Vec<String>)>) -> Self {
        Self { cmds, current_step: 0, output: vec![], done: false, success: false, progress: 0 }
    }
}

// ─── Top-level mode ───────────────────────────────────────────────────────────

pub enum AppMode {
    MainMenu,
    HFSearch(HFSearchState),
    ConvWizard(WizardState),
    JobRunner(JobState),
}
