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
    Convert(String),
    Quantize(String),
    RunModel(String),
    Serve(String),
    Download(String),
    BuildLlama,
}

#[derive(PartialEq)]
enum AppState {
    Menu,
    Inputting { action_idx: usize, prompt: String },
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
        AppAction::Convert(path) => {
            println!("Converting HF model at '{}' to GGUF...", path);
            let mut child = Command::new("python3")
                .arg("./llama.cpp/convert_hf_to_gguf.py")
                .arg(&path)
                .spawn()?;
            child.wait()?;
        }
        AppAction::Quantize(path) => {
            let out_path = format!("{}-Q4_K_M.gguf", path.trim_end_matches(".gguf"));
            println!("Quantizing GGUF model '{}' to '{}'...", path, out_path);
            let mut child = Command::new("./llama.cpp/llama-quantize")
                .arg(&path)
                .arg(&out_path)
                .arg("Q4_K_M")
                .spawn()?;
            child.wait()?;
        }
        AppAction::RunModel(path) => {
            println!("Running model '{}' in interactive mode...", path);
            let mut child = Command::new("./llama.cpp/llama-cli")
                .arg("-m")
                .arg(&path)
                .arg("-cnv")
                .arg("-p")
                .arg("You are a helpful assistant.")
                .spawn()?;
            child.wait()?;
        }
        AppAction::Serve(path) => {
            println!("Serving model '{}' using llama-server on port 8080...", path);
            let mut child = Command::new("./llama.cpp/llama-server")
                .arg("-m")
                .arg(&path)
                .arg("--port")
                .arg("8080")
                .spawn()?;
            child.wait()?;
        }
        AppAction::Download(repo) => {
            println!("Downloading model '{}' via huggingface-cli...", repo);
            let mut child = Command::new("huggingface-cli")
                .arg("download")
                .arg(&repo)
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
            match app.state {
                AppState::Menu => match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Down | KeyCode::Char('j') => app.next(),
                    KeyCode::Up | KeyCode::Char('k') => app.previous(),
                    KeyCode::Enter => {
                        if let Some(selected) = app.menu_state.selected() {
                            match selected {
                                0 => app.state = AppState::Inputting {
                                    action_idx: 0,
                                    prompt: "Enter HF Model Directory:".to_string(),
                                },
                                1 => app.state = AppState::Inputting {
                                    action_idx: 1,
                                    prompt: "Enter GGUF file to quantize:".to_string(),
                                },
                                2 => app.state = AppState::Inputting {
                                    action_idx: 2,
                                    prompt: "Enter GGUF file to run/chat:".to_string(),
                                },
                                3 => app.state = AppState::Inputting {
                                    action_idx: 3,
                                    prompt: "Enter GGUF file to serve (runs on 8080):".to_string(),
                                },
                                4 => app.state = AppState::Inputting {
                                    action_idx: 4,
                                    prompt: "Enter HF Repo ID (e.g. TheBloke/Llama-2-7B-GGUF):".to_string(),
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
                AppState::Inputting { action_idx, prompt: _ } => match key.code {
                    KeyCode::Enter => {
                        let value = app.input.value().to_string();
                        app.log(&format!("Input received: {}", value));
                        match action_idx {
                            0 => app.action_to_exec = Some(AppAction::Convert(value)),
                            1 => app.action_to_exec = Some(AppAction::Quantize(value)),
                            2 => app.action_to_exec = Some(AppAction::RunModel(value)),
                            3 => app.action_to_exec = Some(AppAction::Serve(value)),
                            4 => app.action_to_exec = Some(AppAction::Download(value)),
                            _ => {}
                        }
                        app.input.reset();
                        app.state = AppState::Menu;

                        if app.action_to_exec.is_some() {
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
        AppState::Inputting { prompt, .. } => {
            let input_block = Block::default()
                .borders(Borders::ALL)
                .title(prompt.as_str())
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
