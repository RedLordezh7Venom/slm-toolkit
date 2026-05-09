mod types;
mod graph;
mod tasks;
mod ui;

use color_eyre::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::{io, time::Duration};
use tokio::sync::mpsc;
use tui_input::backend::crossterm::EventHandler;

use types::*;
use graph::{build_commands, dst_fmts_for};

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (tx, mut rx) = mpsc::unbounded_channel::<AppMsg>();
    let mut mode = AppMode::MainMenu;
    let mut menu_idx: usize = 0;

    loop {
        // drain background messages
        while let Ok(msg) = rx.try_recv() {
            match (&mut mode, msg) {
                (AppMode::HFSearch(s), AppMsg::HFResults(r)) => {
                    s.status = format!("{} results — ↑↓ to browse, Enter to download", r.len());
                    s.results = r;
                    s.list_state.select(Some(0));
                }
                (AppMode::HFSearch(s), AppMsg::HFStatus(m)) => s.status = m,
                (AppMode::JobRunner(s), AppMsg::JobLine(l))  => s.output.push(l),
                (AppMode::JobRunner(s), AppMsg::JobStep(n))  => {
                    s.current_step = n;
                    let total = s.cmds.len() as u16;
                    s.progress = ((n as u16) * 100) / total.max(1);
                }
                (AppMode::JobRunner(s), AppMsg::JobDone(ok)) => {
                    s.done = true; s.success = ok; s.progress = 100;
                }
                _ => {}
            }
        }

        terminal.draw(|f| ui::draw(f, &mut mode, menu_idx))?;

        if !event::poll(Duration::from_millis(50))? { continue; }

        if let Event::Key(key) = event::read()? {
            // Global Ctrl-C
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                break;
            }

            match &mut mode {
                // ── Main Menu ─────────────────────────────────────────────
                AppMode::MainMenu => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Up   | KeyCode::Char('k') => { if menu_idx > 0 { menu_idx -= 1; } }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if menu_idx < MENU_ITEMS.len()-1 { menu_idx += 1; }
                    }
                    KeyCode::Enter => match menu_idx {
                        0 => mode = AppMode::ConvWizard(WizardState::new()),
                        1 => mode = AppMode::HFSearch(HFSearchState::new()),
                        _ => break,
                    },
                    _ => {}
                },

                // ── HF Search ─────────────────────────────────────────────
                AppMode::HFSearch(s) => match key.code {
                    KeyCode::Esc => mode = AppMode::MainMenu,
                    KeyCode::Tab => {
                        s.focus = if s.focus == HFFocus::Query { HFFocus::Results } else { HFFocus::Query };
                    }
                    KeyCode::Enter if s.focus == HFFocus::Query => {
                        let q = s.query.value().to_string();
                        if !q.is_empty() {
                            s.status = "Searching…".to_string();
                            s.results.clear();
                            let t = tx.clone();
                            tokio::spawn(async move { tasks::hf_search(q, t).await; });
                            s.focus = HFFocus::Results;
                        }
                    }
                    KeyCode::Enter if s.focus == HFFocus::Results => {
                        if let Some(idx) = s.list_state.selected() {
                            if let Some(repo) = s.results.get(idx).cloned() {
                                s.status = format!("Downloading {}…", repo);
                                let t = tx.clone();
                                tokio::spawn(async move { tasks::hf_download(repo, t).await; });
                            }
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') if s.focus == HFFocus::Results => {
                        let n = s.results.len();
                        if n > 0 {
                            let i = s.list_state.selected().unwrap_or(0);
                            s.list_state.select(Some(if i==0 {n-1} else {i-1}));
                        }
                    }
                    KeyCode::Down | KeyCode::Char('j') if s.focus == HFFocus::Results => {
                        let n = s.results.len();
                        if n > 0 {
                            let i = s.list_state.selected().unwrap_or(0);
                            s.list_state.select(Some((i+1)%n));
                        }
                    }
                    _ if s.focus == HFFocus::Query => { s.query.handle_event(&Event::Key(key)); }
                    _ => {}
                },

                // ── Wizard ────────────────────────────────────────────────
                AppMode::ConvWizard(s) => {
                    let cur = s.step.clone();
                    match key.code {
                        KeyCode::Esc => {
                            let prev = s.prev_step();
                            if prev == WizardStep::SrcFmt && cur == WizardStep::SrcFmt {
                                mode = AppMode::MainMenu;
                            } else {
                                s.step = prev;
                            }
                        }
                        KeyCode::Up | KeyCode::Char('k') => match cur {
                            WizardStep::SrcFmt  => { if s.src_fmt_idx > 0 { s.src_fmt_idx -= 1; } }
                            WizardStep::DstFmt  => { if s.dst_fmt_idx > 0 { s.dst_fmt_idx -= 1; } }
                            WizardStep::OutType => { if s.outtype_idx  > 0 { s.outtype_idx  -= 1; } }
                            WizardStep::QType   => { if s.qtype_idx    > 0 { s.qtype_idx    -= 1; } }
                            _ => {}
                        },
                        KeyCode::Down | KeyCode::Char('j') => match cur {
                            WizardStep::SrcFmt  => {
                                if s.src_fmt_idx < SRC_FMTS.len()-1 { s.src_fmt_idx += 1; }
                                s.dst_fmt_idx = 0;
                            }
                            WizardStep::DstFmt  => {
                                let max = dst_fmts_for(s.src_fmt()).len().saturating_sub(1);
                                if s.dst_fmt_idx < max { s.dst_fmt_idx += 1; }
                            }
                            WizardStep::OutType => {
                                if s.outtype_idx < OUT_TYPES.len()-1 { s.outtype_idx += 1; }
                            }
                            WizardStep::QType   => {
                                if s.qtype_idx < QUANT_TYPES.len()-1 { s.qtype_idx += 1; }
                            }
                            _ => {}
                        },
                        KeyCode::Enter => match cur {
                            WizardStep::Confirm => {
                                // Build & launch job
                                let job = s.to_job();
                                let cmds = build_commands(&job);
                                if cmds.is_empty() {
                                    // no path — stay
                                } else {
                                    let job_state = JobState::new(cmds.clone());
                                    let t = tx.clone();
                                    tokio::spawn(async move { tasks::run_job(cmds, t).await; });
                                    mode = AppMode::JobRunner(job_state);
                                }
                            }
                            WizardStep::SrcPath | WizardStep::OutPath | WizardStep::BaseModel => {
                                let next = s.next_step();
                                s.step = next;
                                s.update_plan();
                            }
                            _ => {
                                let next = s.next_step();
                                s.step = next;
                                s.update_plan();
                            }
                        },
                        _ => match cur {
                            WizardStep::SrcPath   => { s.src_path.handle_event(&Event::Key(key)); }
                            WizardStep::OutPath   => { s.out_path.handle_event(&Event::Key(key)); }
                            WizardStep::BaseModel => { s.base_model.handle_event(&Event::Key(key)); }
                            _ => {}
                        },
                    }
                }

                // ── Job Runner ────────────────────────────────────────────
                AppMode::JobRunner(s) => {
                    if key.code == KeyCode::Esc && s.done {
                        mode = AppMode::MainMenu;
                        menu_idx = 0;
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
