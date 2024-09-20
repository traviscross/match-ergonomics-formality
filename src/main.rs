use core::fmt::{self, Display};
use rustyline::{error::ReadlineError, DefaultEditor};

use match_ergonomics_formality::*;

// # Main

fn help() {
  const TEXT: &str = include_str!("../README.md");
  let mut lines = TEXT.lines();
  for line in &mut lines {
    if line == "# Usage" {
      break;
    }
  }
  for line in &mut lines {
    if line.starts_with("# ") {
      break;
    }
    if line == "```" {
      continue;
    }
    println!("{}", line);
  }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
struct CliConf {
  conf: Conf,
  editable_graph: bool,
  remove_error_edges: bool,
  remove_always_recursive_edges: bool,
  show_graph_source: bool,
}

impl CliConf {
  fn get_mut(&mut self, flag: &str) -> Result<&mut bool, ConfError> {
    if let Ok(flag) = self.conf.get_mut(flag) {
      return Ok(flag);
    }
    Ok(match flag {
      "editable_graph" => &mut self.editable_graph,
      "remove_error_edges" => &mut self.remove_error_edges,
      "remove_always_recursive_edges" => {
        &mut self.remove_always_recursive_edges
      }
      "show_graph_source" => &mut self.show_graph_source,
      _ => return Err(ConfError::UnknownFlag(flag.to_string())),
    })
  }

  fn set(&mut self, flag: &str) -> Result<(), ConfError> {
    if let Ok(()) = self.conf.set(flag) {
      return Ok(());
    }
    let flag = self.get_mut(flag)?;
    *flag = true;
    Ok(())
  }

  fn unset(&mut self, flag: &str) -> Result<(), ConfError> {
    if let Ok(()) = self.conf.unset(flag) {
      return Ok(());
    }
    let flag = self.get_mut(flag)?;
    *flag = false;
    Ok(())
  }
}

impl Display for CliConf {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.conf)?;
    writeln!(f, "| editable_graph: {}", self.editable_graph)?;
    writeln!(f, "| remove_error_edges: {}", self.remove_error_edges)?;
    writeln!(
      f,
      "| remove_always_recursive_edges: {}",
      self.remove_always_recursive_edges
    )?;
    writeln!(f, "| show_graph_source: {}", self.show_graph_source)?;
    Ok(())
  }
}

fn main() {
  let mut rl = DefaultEditor::new().unwrap();
  let mut cliconf =
    CliConf { conf: Conf::rfc2005(), ..<_>::default() };
  let mut cur_stmt = None;
  let mut cur_reduction = None;
  let mut cur_reduction_done = false;
  help();
  loop {
    match rl.readline(">> ") {
      Ok(line) if line.trim() == "help" => {
        rl.add_history_entry(line.as_str()).unwrap();
        help();
      }
      Ok(line) if line.trim_start().starts_with("let ") => {
        rl.add_history_entry(line.as_str()).unwrap();
        let stmt = match LetStmt::from_str(&line) {
          Ok(x) => x,
          Err(err) => {
            print_err(&line, err);
            continue;
          }
        };
        cur_stmt = Some(stmt.clone());
        print!("{:#}", ShowType::from_stmt(cliconf.conf, &stmt));
      }
      Ok(line) if line.trim() == "explain" => {
        let Some(ref stmt) = cur_stmt else {
          println!("ERROR: Provide a statement first.");
          continue;
        };
        let mut r = Reduction::from_stmt(cliconf.conf, stmt.clone());
        if !r.last {
          print!("{}", r);
          r.step();
          cur_reduction = Some(r);
          cur_reduction_done = false;
        } else if !r.is_err() {
          r.apply_dbm();
          print!("{}", r);
          cur_reduction = Some(r);
          cur_reduction_done = true;
        } else {
          println!("# Nothing more to show.");
        }
      }
      Ok(line)
        if line.trim() == "next"
          || line.trim() == "n"
          || line.trim() == "" =>
      {
        let Some(ref mut r) = cur_reduction else {
          if line.trim() != "" {
            println!("ERROR: Call `explain` first.");
          }
          continue;
        };
        if !r.last {
          print!("{}", r);
          r.step();
        } else if !r.is_err() && !cur_reduction_done {
          r.apply_dbm();
          print!("{}", r);
          cur_reduction_done = true;
        } else if line.trim() != "" {
          println!("# Nothing more to show.");
        }
      }
      Ok(line) if line.trim() == "show" => {
        rl.add_history_entry(line.as_str()).unwrap();
        println!("Settings:");
        print!("{cliconf}");
      }
      Ok(line) if line.trim() == "graph" => {
        rl.add_history_entry(line.as_str()).unwrap();
        let mut g = make_graph(cliconf.conf);
        g.simplify_terminal();
        if cliconf.remove_error_edges {
          g.simplify_error();
        }
        if cliconf.remove_always_recursive_edges {
          g.simplify_recuse();
        }
        g.sort_edges();
        if cliconf.show_graph_source {
          println!("{}", g);
        }
        if cliconf.editable_graph {
          print!("{}", Mermaid::new(MermaidLinkKind::Edit, &g));
        } else {
          print!("{}", Mermaid::new(MermaidLinkKind::View, &g));
        }
        println!();
      }
      Ok(line) if line.trim() == "graph svg" => {
        rl.add_history_entry(line.as_str()).unwrap();
        let mut g = make_graph(cliconf.conf);
        g.simplify_terminal();
        if cliconf.remove_error_edges {
          g.simplify_error();
        }
        if cliconf.remove_always_recursive_edges {
          g.simplify_recuse();
        }
        if cliconf.show_graph_source {
          println!("{}", g);
        }
        println!("{}", Mermaid::new(MermaidLinkKind::Img("svg"), &g));
      }
      Ok(line) if line.trim().starts_with("set ") => {
        rl.add_history_entry(line.as_str()).unwrap();
        let flag = &line.trim()["set ".len()..];
        if let Err(e) = cliconf.set(flag) {
          println!("# ERROR: {e}");
        }
      }
      Ok(line) if line.trim().starts_with("unset ") => {
        rl.add_history_entry(line.as_str()).unwrap();
        let flag = &line.trim()["unset ".len()..];
        if let Err(e) = cliconf.unset(flag) {
          println!("# ERROR: {e}");
        }
      }
      Ok(line) => {
        rl.add_history_entry(line.as_str()).unwrap();
        println!("# ERROR: Unrecognized command: {}", line);
      }
      Err(ReadlineError::Eof) => {
        println!("# ctrl-d received");
        break;
      }
      Err(ReadlineError::Interrupted) => {
        println!("# ctrl-c received");
        break;
      }
      Err(e) => {
        println!("# ERROR: {}", e);
        break;
      }
    }
  }
}
