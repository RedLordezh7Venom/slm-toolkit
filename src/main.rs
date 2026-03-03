use color_eyre::Result;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};
use std::io;
use std::process::Command;
use tui_input::backend::crossterm::EventHandler;
use tui_input::Input;

#[derive(PartialEq, Clone)]
enum AppAction {
    Convert { dir: String, outtype: String, outfile: String },
    Quantize { file: String, outfile: String, qtype: String },
    RunModel { file: String, prompt: String },
    Serve { file: String, port: String },
    Download { repo: String },
    BuildLlama,
}

#[derive(PartialEq, Clone)]
enum AppState {
    Menu,
    Inputting { action_idx: usize, step: usize, prompts: Vec<&'static str>, answers: Vec<String> },
}

struct App {
    state: AppState,
    menu_state: ListState,
    items: Vec<&'static str>,
    input: Input,
    logs: Vec<String>,
    action_to_exec: Option<AppAction>,
}

impl App {
    fn new() -> App {
        let mut menu_state = ListState::default();
        menu_state.select(Some(0));
        App {
            state: AppState::Menu,
            menu_state,
            items: vec![
                "[1] Convert HuggingFace Model to GGUF",
                "[2] Quantize GGUF Model",
                "[3] Run / Chat with Model",
                "[4] Serve Model (API)",
                "[5] Download Model from HuggingFace",
                "[6] Setup / Build llama.cpp",
                "[7] Exit",
            ],
            input: Input::default(),
            logs: vec!["Welcome to SLM Toolkit!".to_string()],
            action_to_exec: None,
        }
    }

    fn next(&mut self) {
        let i = match self.menu_state.selected() {
            Some(i) => {
                if i >= self.items.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.menu_state.select(Some(i));
    }

    fn previous(&mut self) {
        let i = match self.menu_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.items.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.menu_state.select(Some(i));
    }

    fn log(&mut self, text: &str) {
        self.logs.push(text.to_string());
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    
    let mut app = App::new();

    loop {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        app.action_to_exec = None;
        let res = run_app(&mut terminal, &mut app).await;

        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen
        )?;
        terminal.show_cursor()?;

        if let Err(err) = res {
            println!("{:?}", err);
            break;
        }

        if let Some(action) = app.action_to_exec.take() {
            handle_action(action).await?;
            println!("\nPress ENTER to return to the menu...");
            let mut line = String::new();
            let _ = io::stdin().read_line(&mut line);
        } else {
            // User exited
            break;
        }
    }

    Ok(())
}

async fn handle_action(action: AppAction) -> Result<()> {
    match action {
        AppAction::Convert { dir, outtype, outfile } => {
            if dir.trim().is_empty() {
                println!("Error: Input directory cannot be empty!");
                return Ok(());
            }
            let out_arg = if outtype.trim().is_empty() { "f16".to_string() } else { outtype.trim().to_string() };
            let outfile_arg = if outfile.trim().is_empty() { "".to_string() } else { format!("--outfile {}", outfile.trim()) };
            
            println!("Converting HF model at '{}' to GGUF ({}) {}...", dir, out_arg, outfile_arg);
            let mut child = Command::new("bash")
                .arg("-c")
                .arg(format!("source ./llama.cpp/.venv/bin/activate && python3 ./llama.cpp/convert_hf_to_gguf.py {} --outtype {} {}", dir, out_arg, outfile_arg))
                .spawn()?;
            child.wait()?;
        }
        AppAction::Quantize { file, outfile, qtype } => {
            if file.trim().is_empty() {
                println!("Error: Input GGUF cannot be empty!");
                return Ok(());
            }
            let out_arg = if outfile.trim().is_empty() { format!("{}-quantized.gguf", file.trim_end_matches(".gguf")) } else { outfile.trim().to_string() };
            let q_arg = if qtype.trim().is_empty() { "Q4_K_M".to_string() } else { qtype.trim().to_string() };
            
            println!("Quantizing GGUF model '{}' to '{}' using {}...", file, out_arg, q_arg);
            let mut child = Command::new("./llama.cpp/llama-quantize")
                .arg(&file)
                .arg(&out_arg)
                .arg(&q_arg)
                .spawn()?;
            child.wait()?;
        }
        AppAction::RunModel { file, prompt } => {
            if file.trim().is_empty() {
                println!("Error: Input GGUF cannot be empty!");
                return Ok(());
            }
            let sys_prompt = if prompt.trim().is_empty() { "You are a helpful assistant." } else { prompt.trim() };
            println!("Running model '{}' in interactive mode...", file);
            let mut child = Command::new("./llama.cpp/llama-cli")
                .arg("-m")
                .arg(&file)
                .arg("-cnv")
                .arg("-p")
                .arg(sys_prompt)
                .spawn()?;
            child.wait()?;
        }
        AppAction::Serve { file, port } => {
            if file.trim().is_empty() {
                println!("Error: Input GGUF cannot be empty!");
                return Ok(());
            }
            let p_arg = if port.trim().is_empty() { "8080" } else { port.trim() };
            println!("Serving model '{}' using llama-server on port {}...", file, p_arg);
            let mut child = Command::new("./llama.cpp/llama-server")
                .arg("-m")
                .arg(&file)
                .arg("--port")
                .arg(p_arg)
                .spawn()?;
            child.wait()?;
        }
        AppAction::Download { repo } => {
            if repo.trim().is_empty() {
                println!("Error: Repo ID cannot be empty!");
                return Ok(());
            }
            println!("Downloading model(s) via huggingface-cli for '{}'...", repo);
            let mut child = Command::new("bash")
                .arg("-c")
                .arg(format!("huggingface-cli download {}", repo.trim()))
                .spawn()?;
            child.wait()?;
        }
        AppAction::BuildLlama => {
            println!("Building llama.cpp using make...");
            let mut child = Command::new("make")
                .current_dir("./llama.cpp")
                .spawn()?;
            child.wait()?;
        }
    }
    Ok(())
}

async fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> color_eyre::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if let Event::Key(key) = event::read()? {
            let current_state = app.state.clone();
            match current_state {
                AppState::Menu => match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Down | KeyCode::Char('j') => app.next(),
                    KeyCode::Up | KeyCode::Char('k') => app.previous(),
                    KeyCode::Enter => {
                        if let Some(selected) = app.menu_state.selected() {
                            match selected {
                                0 => app.state = AppState::Inputting {
                                    action_idx: 0, step: 0, answers: vec![],
                                    prompts: vec!["Enter HF Model Directory:", "Enter Output Format (e.g. f16, q8_0) [default: f16]:", "Enter Output File Path (optional) [default: auto]:"],
                                },
                                1 => app.state = AppState::Inputting {
                                    action_idx: 1, step: 0, answers: vec![],
                                    prompts: vec!["Enter GGUF file to quantize:", "Enter Output File (optional) [default: auto]:", "Enter Quantization Type (e.g. Q4_K_M) [default: Q4_K_M]:"],
                                },
                                2 => app.state = AppState::Inputting {
                                    action_idx: 2, step: 0, answers: vec![],
                                    prompts: vec!["Enter GGUF file to run/chat:", "Enter System Prompt [default: You are a helpful assistant.]:"],
                                },
                                3 => app.state = AppState::Inputting {
                                    action_idx: 3, step: 0, answers: vec![],
                                    prompts: vec!["Enter GGUF file to serve:", "Enter Port [default: 8080]:"],
                                },
                                4 => app.state = AppState::Inputting {
                                    action_idx: 4, step: 0, answers: vec![],
                                    prompts: vec!["Enter HF Repo ID (e.g. TheBloke/Llama-2-7B-GGUF):"],
                                },
                                5 => {
                                    app.action_to_exec = Some(AppAction::BuildLlama);
                                    return Ok(());
                                }
                                6 => return Ok(()),
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                },
                AppState::Inputting { action_idx, step, prompts, mut answers } => match key.code {
                    KeyCode::Enter => {
                        let value = app.input.value().to_string();
                        app.log(&format!("Input step {} received: {}", step, value));
                        let a_idx = action_idx;
                        let s = step;
                        let prompts_clone = prompts;
                        let mut answers_clone = answers;
                        answers_clone.push(value);
                        
                        app.input.reset();
                        
                        if s + 1 < prompts_clone.len() {
                            app.state = AppState::Inputting {
                                action_idx: a_idx,
                                step: s + 1,
                                prompts: prompts_clone,
                                answers: answers_clone,
                            };
                        } else {
                            match a_idx {
                                0 => app.action_to_exec = Some(AppAction::Convert { dir: answers_clone[0].clone(), outtype: answers_clone[1].clone(), outfile: answers_clone[2].clone() }),
                                1 => app.action_to_exec = Some(AppAction::Quantize { file: answers_clone[0].clone(), outfile: answers_clone[1].clone(), qtype: answers_clone[2].clone() }),
                                2 => app.action_to_exec = Some(AppAction::RunModel { file: answers_clone[0].clone(), prompt: answers_clone[1].clone() }),
                                3 => app.action_to_exec = Some(AppAction::Serve { file: answers_clone[0].clone(), port: answers_clone[1].clone() }),
                                4 => app.action_to_exec = Some(AppAction::Download { repo: answers_clone[0].clone() }),
                                _ => {}
                            }
                            app.state = AppState::Menu;
                            return Ok(());
                        }
                    }
                    KeyCode::Esc => {
                        app.input.reset();
                        app.state = AppState::Menu;
                    }
                    _ => {
                        app.input.handle_event(&Event::Key(key));
                    }
                },
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(7)].as_ref())
        .split(f.area());

    let title_block = Block::default()
        .borders(Borders::ALL)
        .title(" SLM Toolkit ")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    
    let title = Paragraph::new("Simplify Small Language Models operations easily!")
        .block(title_block)
        .alignment(Alignment::Center);
    f.render_widget(title, chunks[0]);

    match &app.state {
        AppState::Menu => {
            let items: Vec<ListItem> = app
                .items
                .iter()
                .map(|i| {
                    ListItem::new(vec![Line::from(Span::styled(
                        *i,
                        Style::default().fg(Color::White),
                    ))])
                })
                .collect();

            let list = List::new(items)
                .block(Block::default().borders(Borders::ALL).title(" Main Menu "))
                .highlight_style(
                    Style::default()
                        .bg(Color::Cyan)
                        .fg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                )
                .highlight_symbol(">> ");

            f.render_stateful_widget(list, chunks[1], &mut app.menu_state);
        }
        AppState::Inputting { step, prompts, .. } => {
            let p = prompts[*step];
            let input_block = Block::default()
                .borders(Borders::ALL)
                .title(p)
                .style(Style::default().fg(Color::Yellow));

            let input_val = app.input.value();
            let input_widget = Paragraph::new(input_val).block(input_block);
            
            let area = centered_rect(60, 20, chunks[1]);
            f.render_widget(input_widget, area);
            
            f.set_cursor_position(Position::new(
                area.x + app.input.visual_cursor() as u16 + 1,
                area.y + 1,
            ));
        }
    }

    let logs_text: Vec<Line> = app.logs.iter().rev().take(5).map(|l| Line::from(l.as_str())).collect();
    let logs_block = Block::default().borders(Borders::ALL).title(" Logs ");
    let logs_widget = Paragraph::new(logs_text).block(logs_block);
    f.render_widget(logs_widget, chunks[2]);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
