use ratatui::{prelude::*, widgets::*};
use crate::types::*;
use crate::graph::{dst_fmts_for, step_label, resolve_steps};

const CYAN:    Color = Color::Cyan;
const YELLOW:  Color = Color::Yellow;
const GREEN:   Color = Color::Green;
const RED:     Color = Color::Red;
const DARK:    Color = Color::Rgb(18, 18, 30);
const PANEL:   Color = Color::Rgb(30, 41, 59);
const DIM:     Color = Color::Rgb(100, 116, 139);
const BRIGHT:  Color = Color::White;

fn title_block(title: &str, color: Color) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color))
        .title(Span::styled(format!(" {title} "), Style::default().fg(color).add_modifier(Modifier::BOLD)))
}

fn highlight_style() -> Style {
    Style::default().bg(CYAN).fg(Color::Black).add_modifier(Modifier::BOLD)
}

// ─── Top-level dispatcher ─────────────────────────────────────────────────────

pub fn draw(f: &mut Frame, mode: &mut AppMode, menu_idx: usize) {
    f.render_widget(Block::default().style(Style::default().bg(DARK)), f.area());

    match mode {
        AppMode::MainMenu            => draw_menu(f, menu_idx),
        AppMode::HFSearch(s)         => draw_hf(f, s),
        AppMode::ConvWizard(s)       => draw_wizard(f, s),
        AppMode::JobRunner(s)        => draw_job(f, s),
    }
}

// ─── Main Menu ────────────────────────────────────────────────────────────────

fn draw_menu(f: &mut Frame, idx: usize) {
    let area = f.area();
    let v = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(MENU_ITEMS.len() as u16 + 2),
        Constraint::Min(0),
        Constraint::Length(1),
    ]).split(area);

    // Banner
    let banner = Paragraph::new(vec![
        Line::from(Span::styled("⚡  SLM Toolkit", Style::default().fg(CYAN).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled("   Model Conversion Suite · Powered by llama.cpp",
            Style::default().fg(DIM))),
    ])
    .alignment(Alignment::Center)
    .block(title_block("", CYAN));
    f.render_widget(banner, v[0]);

    // Menu list
    let items: Vec<ListItem> = MENU_ITEMS.iter().enumerate().map(|(i, &item)| {
        let style = if i == idx {
            Style::default().fg(Color::Black).bg(CYAN).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(BRIGHT)
        };
        ListItem::new(Line::from(Span::styled(format!("  {item}  "), style)))
    }).collect();
    let list = List::new(items).block(title_block("Main Menu", CYAN));
    f.render_widget(list, centered_rect(60, v[1]));

    // Footer hint
    let footer = Paragraph::new("↑ ↓  j k  navigate    Enter  select    q  quit")
        .style(Style::default().fg(DIM))
        .alignment(Alignment::Center);
    f.render_widget(footer, v[3]);
}

// ─── HF Search ───────────────────────────────────────────────────────────────

fn draw_hf(f: &mut Frame, s: &mut HFSearchState) {
    let area = f.area();
    let v = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(3),
        Constraint::Length(1),
    ]).split(area);

    let title = Paragraph::new("Search HuggingFace Hub — results sorted by downloads")
        .alignment(Alignment::Center)
        .block(title_block("HuggingFace Browser", CYAN));
    f.render_widget(title, v[0]);

    // Query input
    let q_border_color = if s.focus == HFFocus::Query { YELLOW } else { DIM };
    let q_block = title_block("Search query  (Enter to search)", q_border_color);
    let q_widget = Paragraph::new(s.query.value()).block(q_block);
    f.render_widget(q_widget, v[1]);
    if s.focus == HFFocus::Query {
        f.set_cursor_position(Position::new(v[1].x + s.query.visual_cursor() as u16 + 1, v[1].y + 1));
    }

    // Results list
    let r_border_color = if s.focus == HFFocus::Results { YELLOW } else { DIM };
    let items: Vec<ListItem> = s.results.iter().map(|r| {
        ListItem::new(Line::from(Span::styled(format!("  {r}"), Style::default().fg(BRIGHT))))
    }).collect();
    let list = List::new(items)
        .block(title_block("Results  (Enter to download)", r_border_color))
        .highlight_style(highlight_style())
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, v[2], &mut s.list_state);

    // Status bar
    let status = Paragraph::new(s.status.as_str())
        .style(Style::default().fg(DIM))
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(PANEL)));
    f.render_widget(status, v[3]);

    // Footer
    let footer = Paragraph::new("Tab  switch focus    ↑↓  navigate results    Enter  search / download    Esc  back")
        .style(Style::default().fg(DIM)).alignment(Alignment::Center);
    f.render_widget(footer, v[4]);
}

// ─── Wizard ───────────────────────────────────────────────────────────────────

fn draw_wizard(f: &mut Frame, s: &mut WizardState) {
    let area = f.area();
    let v = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(6),
        Constraint::Length(1),
    ]).split(area);

    // Title
    let step_name = match s.step {
        WizardStep::SrcFmt    => "Step 1 · Source Format",
        WizardStep::SrcPath   => "Step 2 · Source Path",
        WizardStep::DstFmt    => "Step 3 · Target Format",
        WizardStep::OutType   => "Step 4 · Output Type (for GGUF step)",
        WizardStep::QType     => "Step 5 · Quantization Level",
        WizardStep::OutPath   => "Step 6 · Output Path",
        WizardStep::BaseModel => "Step 7 · Base Model Path (LoRA)",
        WizardStep::Confirm   => "Confirm & Start",
    };
    let title = Paragraph::new(step_name)
        .alignment(Alignment::Center)
        .block(title_block("Conversion Wizard", CYAN));
    f.render_widget(title, v[0]);

    // Step content
    match s.step {
        WizardStep::SrcFmt => {
            let opts: Vec<(String, String)> = SRC_FMTS.iter().map(|&(v, l)| (v.to_string(), l.to_string())).collect();
            draw_select_list(f, v[1], "Select source format:", &opts, s.src_fmt_idx);
        },
        WizardStep::SrcPath => draw_input(
            f, v[1],
            &format!("Path / HF repo ID  [{} selected]", s.src_fmt()),
            &s.src_path,
        ),
        WizardStep::DstFmt => {
            let opts: Vec<(String, String)> = dst_fmts_for(s.src_fmt()).iter().map(|&(v, l)| (v.to_string(), l.to_string())).collect();
            draw_select_list(f, v[1], "Select target format:", &opts, s.dst_fmt_idx);
        }
        WizardStep::OutType => {
            let pairs: Vec<(String, String)> = crate::types::OUT_TYPES.iter().map(|&t| (t.to_string(), t.to_string())).collect();
            draw_select_list(f, v[1], "Output format (for HF→GGUF):", &pairs, s.outtype_idx);
        }
        WizardStep::QType => {
            let pairs: Vec<(String, String)> = crate::types::QUANT_TYPES.iter().map(|&q| (q.to_string(), q.to_string())).collect();
            draw_select_list(f, v[1], "Quantization type:", &pairs, s.qtype_idx);
        }
        WizardStep::OutPath => draw_input(f, v[1], "Output path (leave blank = auto):", &s.out_path),
        WizardStep::BaseModel => draw_input(f, v[1], "Base model directory (optional):", &s.base_model),
        WizardStep::Confirm => {
            let steps = resolve_steps(s.src_fmt(), s.dst_fmt()).unwrap_or_default();
            let mut lines = vec![
                Line::from(Span::styled("  Ready to convert!", Style::default().fg(GREEN).add_modifier(Modifier::BOLD))),
                Line::from(""),
                Line::from(Span::styled(format!("  {} → {}", s.src_fmt(), s.dst_fmt()), Style::default().fg(CYAN))),
                Line::from(Span::styled(format!("  Source:  {}", s.src_path.value()), Style::default().fg(DIM))),
            ];
            for (i, step) in steps.iter().enumerate() {
                lines.push(Line::from(Span::styled(
                    format!("  {}. {}", i+1, step_label(step)), Style::default().fg(YELLOW),
                )));
            }
            let p = Paragraph::new(lines).block(title_block("Conversion Plan", GREEN));
            f.render_widget(p, v[1]);
        }
    }

    // Plan preview at bottom
    let plan_lines: Vec<Line> = if s.plan.is_empty() {
        vec![Line::from(Span::styled("  Select source & target format to preview steps", Style::default().fg(DIM)))]
    } else {
        s.plan.iter().map(|l| {
            let color = if l.contains('✗') { RED } else { YELLOW };
            Line::from(Span::styled(l.as_str(), Style::default().fg(color)))
        }).collect()
    };
    let plan = Paragraph::new(plan_lines).block(title_block("Conversion Plan Preview", DIM));
    f.render_widget(plan, v[2]);

    // Footer
    let hint = match s.step {
        WizardStep::Confirm => "Enter  start conversion    Esc  back",
        WizardStep::SrcPath | WizardStep::OutPath | WizardStep::BaseModel =>
            "Enter  confirm    Esc  back",
        _ => "↑ ↓  select    Enter  confirm    Esc  back",
    };
    let footer = Paragraph::new(hint).style(Style::default().fg(DIM)).alignment(Alignment::Center);
    f.render_widget(footer, v[3]);
}

fn draw_select_list(f: &mut Frame, area: Rect, prompt: &str, items: &[(impl AsRef<str>, impl AsRef<str>)], selected: usize) {
    let list_items: Vec<ListItem> = items.iter().enumerate().map(|(i, (_, label))| {
        let style = if i == selected { highlight_style() } else { Style::default().fg(BRIGHT) };
        ListItem::new(Line::from(Span::styled(format!("  {}", label.as_ref()), style)))
    }).collect();
    let mut state = ListState::default();
    state.select(Some(selected));
    let list = List::new(list_items)
        .block(title_block(prompt, YELLOW))
        .highlight_style(highlight_style());
    f.render_stateful_widget(list, area, &mut state);
}

fn draw_input(f: &mut Frame, area: Rect, prompt: &str, input: &tui_input::Input) {
    let block = title_block(prompt, YELLOW);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let p = Paragraph::new(input.value()).style(Style::default().fg(BRIGHT));
    f.render_widget(p, inner);
    f.set_cursor_position(Position::new(inner.x + input.visual_cursor() as u16, inner.y));
}

// ─── Job Runner ───────────────────────────────────────────────────────────────

fn draw_job(f: &mut Frame, s: &JobState) {
    let area = f.area();
    let v = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ]).split(area);

    // Status
    let (msg, color) = if s.done {
        if s.success { ("✓  All steps completed!", GREEN) } else { ("✗  Conversion failed", RED) }
    } else {
        let step_label = s.cmds.get(s.current_step).map(|(l,_)| l.as_str()).unwrap_or("…");
        (step_label, YELLOW)
    };
    let status = Paragraph::new(Span::styled(msg, Style::default().fg(color).add_modifier(Modifier::BOLD)))
        .alignment(Alignment::Center)
        .block(title_block("Status", color));
    f.render_widget(status, v[0]);

    // Progress bar
    let label = format!("{} / {} steps", s.current_step, s.cmds.len());
    let gauge = Gauge::default()
        .block(title_block(&label, DIM))
        .gauge_style(Style::default().fg(CYAN).bg(PANEL))
        .ratio(s.progress as f64 / 100.0)
        .label(Span::styled(format!("{}%", s.progress), Style::default().fg(BRIGHT)));
    f.render_widget(gauge, v[1]);

    // Log output — show last N lines
    let height = v[2].height.saturating_sub(2) as usize;
    let start = s.output.len().saturating_sub(height);
    let log_lines: Vec<Line> = s.output[start..].iter().map(|l| {
        let color = if l.starts_with("✓") { GREEN }
            else if l.starts_with("✗") || l.starts_with("[err]") { RED }
            else if l.starts_with("───") { CYAN }
            else { BRIGHT };
        Line::from(Span::styled(l.as_str(), Style::default().fg(color)))
    }).collect();
    let log = Paragraph::new(log_lines).block(title_block("Output", DIM));
    f.render_widget(log, v[2]);

    // Footer
    let hint = if s.done { "Esc  back to menu" } else { "Running…  Ctrl-C to force quit" };
    let footer = Paragraph::new(hint).style(Style::default().fg(DIM)).alignment(Alignment::Center);
    f.render_widget(footer, v[3]);
}

// ─── Utility ─────────────────────────────────────────────────────────────────

fn centered_rect(percent_x: u16, area: Rect) -> Rect {
    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ]).split(area)[1]
}
