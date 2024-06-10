#![allow(unused)]
#![warn(unreachable_code)]

use core::{
  fmt::{self, Display, Write},
  mem::swap,
  result,
};
use std::{
  collections::{HashMap, HashSet},
  io,
};

// # Extension traits

trait SliceExt {
  fn split_at_opt(&self, mid: usize) -> Option<(&Self, &Self)>;
}

impl<T> SliceExt for [T] {
  fn split_at_opt(&self, mid: usize) -> Option<(&Self, &Self)> {
    self.get(..mid).and_then(|x| Some((x, self.get(mid..)?)))
  }
}

trait StrExt {
  fn split_at_opt(&self, mid: usize) -> Option<(&str, &str)>;
  fn split_prefix<'a>(
    &'a self,
    prefix: &'a str,
  ) -> Option<(&'a str, &'a str)>;
  fn strip_prefix2(&self, prefix: &str) -> Option<(usize, &str)>;
}

impl StrExt for str {
  fn split_at_opt(&self, mid: usize) -> Option<(&str, &str)> {
    self.get(..mid).and_then(|x| Some((x, self.get(mid..)?)))
  }

  fn split_prefix<'a>(
    &'a self,
    prefix: &'a str,
  ) -> Option<(&'a str, &'a str)> {
    self.starts_with(prefix).then(|| (prefix, &self[prefix.len()..]))
  }

  fn strip_prefix2(&self, prefix: &str) -> Option<(usize, &str)> {
    self.starts_with(prefix).then(|| {
      let len = prefix.len();
      (len, &self[len..])
    })
  }
}

// # Shared

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Span {
  start: usize,
  end: usize,
}

impl Span {
  fn new(start: usize, len: usize) -> Self {
    Self { start, end: start + len }
  }
  fn nop() -> Self {
    Self { start: 0, end: 0 }
  }
  fn from2(span1: Span, span2: Span) -> Self {
    Self { start: span1.start, end: span2.end }
  }
  fn len(&self) -> usize {
    self.end - self.start
  }
  fn extend(&self, end: usize) -> Self {
    Self { start: self.start, end }
  }
  fn extend_until(&self, span: Span) -> Self {
    Self { start: self.start, end: span.start }
  }
  fn extend_across(&self, span: Span) -> Self {
    Self { start: self.start, end: span.end }
  }
  fn with_len(&self, len: usize) -> Self {
    Self { start: self.start, end: self.start + len }
  }
  fn len_until(&self, span: Span) -> usize {
    assert!(span.start >= self.end);
    span.start - self.end
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Ident {
  name: String,
  span: Span,
}

impl Ident {
  fn new(name: String, span: Span) -> Self {
    Self { name, span }
  }
  fn len(&self) -> usize {
    self.name.len()
  }
}

impl Display for Ident {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.name)
  }
}

// # Error handling

#[derive(Debug)]
enum Error {
  Unexpected(Span),
  Expected(Span, TokenKind),
  Unexpected_,
  UnterminatedSlice,
}

fn print_err(xs: &str, err: Error) {
  match err {
    Error::Unexpected(span) => {
      println!("error: unexpected input");
      println!("| {}", xs);
      print!("| ");
      for _ in 0..span.start {
        print!(" ");
      }
      for _ in 0..span.len().max(1) {
        print!("^");
      }
      println!(" unexpected");
    }
    Error::Expected(span, kind) => {
      println!("error: expected {kind:?}");
      println!("| {}", xs);
      print!("| ");
      for _ in 0..span.start {
        print!(" ");
      }
      for _ in 0..span.len().max(1) {
        print!("^");
      }
      println!(" expected {kind:?}");
    }
    Error::Unexpected_ => {
      println!("error: unexpected input");
    }
    Error::UnterminatedSlice => {
      println!("error: unterminated slice");
    }
  }
}

// # Lexing

#[derive(Clone, Debug, Eq, PartialEq)]
struct Token {
  kind: TokenKind,
  span: Span,
}

impl Token {
  fn new(kind: TokenKind, idx: usize) -> Self {
    let len = kind.len();
    Self { kind, span: Span::new(idx, len) }
  }
  fn new_span(kind: TokenKind, span: Span) -> Self {
    Self { kind, span }
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TokenKind {
  Amp,
  AmpMut,
  Ref,
  RefMut,
  Mut,
  ParenStart,
  ParenEnd,
  SliceEnd,
  SliceStart,
  Equals,
  Let,
  Semicolon,
  Binding(Ident),
  Type(Ident),
}

impl TokenKind {
  fn len(&self) -> usize {
    match self {
      TokenKind::Amp => const { "&".len() },
      TokenKind::AmpMut => const { "&mut".len() },
      TokenKind::Ref => const { "ref".len() },
      TokenKind::RefMut => const { "ref mut".len() },
      TokenKind::Mut => const { "mut".len() },
      TokenKind::ParenStart => const { "(".len() },
      TokenKind::ParenEnd => const { ")".len() },
      TokenKind::SliceEnd => const { "[".len() },
      TokenKind::SliceStart => const { "[".len() },
      TokenKind::Equals => const { "=".len() },
      TokenKind::Let => const { "let".len() },
      TokenKind::Semicolon => const { ";".len() },
      TokenKind::Binding(ident) => ident.len(),
      TokenKind::Type(ident) => ident.len(),
    }
  }
}

fn lex(mut xs: &str) -> Result<Vec<Token>, Error> {
  use TokenKind::*;
  let mut ys = Vec::new();
  let mut idx: usize = 0;
  loop {
    if xs.is_empty() {
      break Ok(ys);
    }
    if let Some((len, xs_)) = xs.strip_prefix2("&") {
      ys.push(Token::new(Amp, idx));
      (xs, idx) = (xs_, idx + len);
    } else if let Some((len, xs_)) = xs.strip_prefix2("ref") {
      ys.push(Token::new(Ref, idx));
      (xs, idx) = (xs_, idx + len);
    } else if let Some((len, xs_)) = xs.strip_prefix2("mut") {
      if let Some(&Token { kind: Amp, span }) = ys.last() {
        _ = ys.pop();
        ys.push(Token::new_span(AmpMut, span.extend(idx + len)));
      } else if let Some(&Token { kind: Ref, span }) = ys.last() {
        _ = ys.pop();
        ys.push(Token::new_span(RefMut, span.extend(idx + len)));
      } else {
        ys.push(Token::new(Mut, idx));
      }
      (xs, idx) = (xs_, idx + len);
    } else if let Some((len, xs_)) = xs.strip_prefix2("(") {
      ys.push(Token::new(ParenStart, idx));
      (xs, idx) = (xs_, idx + len);
    } else if let Some((len, xs_)) = xs.strip_prefix2(")") {
      ys.push(Token::new(ParenEnd, idx));
      (xs, idx) = (xs_, idx + len);
    } else if let Some((len, xs_)) = xs.strip_prefix2("[") {
      ys.push(Token::new(SliceStart, idx));
      (xs, idx) = (xs_, idx + len);
    } else if let Some((len, xs_)) = xs.strip_prefix2("]") {
      ys.push(Token::new(SliceEnd, idx));
      (xs, idx) = (xs_, idx + len);
    } else if let Some((len, xs_)) = xs.strip_prefix2("=") {
      ys.push(Token::new(Equals, idx));
      (xs, idx) = (xs_, idx + len);
    } else if let Some((len, xs_)) = xs.strip_prefix2("let") {
      ys.push(Token::new(Let, idx));
      (xs, idx) = (xs_, idx + len);
    } else if let Some((len, xs_)) = xs.strip_prefix2(";") {
      ys.push(Token::new(Semicolon, idx));
      (xs, idx) = (xs_, idx + len);
    } else {
      let mut cs = xs.chars().peekable();
      let p = cs.peek().unwrap();
      if p.is_alphabetic() {
        if p.is_ascii_uppercase() {
          let s = cs
            .take_while(|x| {
              x.is_ascii_uppercase() || x.is_ascii_digit()
            })
            .collect::<String>();
          let len = s.len();
          let span = Span::new(idx, len);
          ys.push(Token::new_span(Type(Ident::new(s, span)), span));
          (xs, idx) = (&xs[len..], idx + len);
          continue;
        } else if p.is_ascii_lowercase() {
          let s = cs
            .take_while(|x| {
              x.is_ascii_lowercase() || x.is_ascii_digit()
            })
            .collect::<String>();
          let len = s.len();
          let span = Span::new(idx, len);
          ys.push(Token::new_span(
            Binding(Ident::new(s, span)),
            span,
          ));
          (xs, idx) = (&xs[len..], idx + len);
          continue;
        }
      } else if p.is_ascii_whitespace() {
        (xs, idx) = (&xs[1..], idx + 1);
      } else {
        return Err(Error::Unexpected(Span::new(idx, 1)));
      }
    }
  }
}

fn unlex(xs: Vec<Token>) -> String {
  let mut ys = String::new();
  for x in xs {
    use TokenKind::*;
    match x.kind {
      Amp => ys.push_str("&"),
      AmpMut => ys.push_str("&mut "),
      Ref => ys.push_str("ref "),
      RefMut => ys.push_str("ref mut "),
      Mut => ys.push_str("mut "),
      ParenEnd => ys.push_str(")"),
      ParenStart => ys.push_str("("),
      SliceEnd => ys.push_str("]"),
      SliceStart => ys.push_str("["),
      Equals => ys.push_str(" = "),
      Let => ys.push_str("let "),
      Semicolon => ys.push_str(";"),
      Binding(ident) => ys.push_str(&ident.name),
      Type(ident) => ys.push_str(&ident.name),
    }
  }
  ys
}

#[test]
fn test_lex() {
  let xs =
    "let &[&mut [&&mut [&ref mut xxx]]] = &[&mut [&&mut [&T]]];";
  let ys = unlex(lex(xs).unwrap());
  assert_eq!(xs, ys);
  let xs = "let &[&mut [&&mut [&(mut xxx)]]] = &[&mut [&&mut [&T]]];";
  let ys = unlex(lex(xs).unwrap());
  assert_eq!(xs, ys);
  let xs = "let &([&mut [&&mut [&x]]]) = &([&mut [&&mut [&TTT]]]);";
  let ys = unlex(lex(xs).unwrap());
  assert_eq!(xs, ys);
}

// # Parsing interface

#[derive(Clone, Debug)]
struct Ctx<'x> {
  rem: &'x [Token],
  last_idx: usize,
}

impl<'x> Iterator for Ctx<'x> {
  type Item = &'x Token;
  fn next(&mut self) -> Option<Self::Item> {
    if let Some(x) = self.rem.first() {
      self.rem(&self.rem[1..]);
      Some(x)
    } else {
      None
    }
  }
}

impl<'x> Ctx<'x> {
  fn new(rem: &'x [Token]) -> Self {
    let mut last_idx = 0;
    if let Some(Token { span, .. }) = rem.first() {
      last_idx = span.start;
    }
    Ctx { rem, last_idx }
  }

  fn rem(&mut self, rem: &'x [Token]) {
    if let Some(Token { span, .. }) = rem.first() {
      self.last_idx = span.end;
    }
    self.rem = rem;
  }

  fn next_token(&mut self) -> Result<(&TokenKind, Span), Error> {
    if let Some(x) = self.rem.first() {
      self.rem(&self.rem[1..]);
      Ok((&x.kind, x.span))
    } else {
      Err(Error::Unexpected(Span::new(self.last_idx, 0)))
    }
  }

  fn expect(&mut self, x: &[TokenKind]) -> Result<Span, Error> {
    if x.is_empty() {
      panic!("expected zero tokens");
    }
    if self.rem.is_empty() {
      return Err(Error::Unexpected(Span::new(self.last_idx, 0)));
    }
    for (x, y) in self.rem.iter().zip(x.iter()) {
      if &x.kind != y {
        return Err(Error::Expected(x.span, y.clone()));
      }
    }
    let span =
      Span::from2(self.rem[0].span, self.rem[x.len() - 1].span);
    self.rem(&self.rem[x.len()..]);
    Ok(span)
  }

  fn peek_expect(&self) -> Result<(&TokenKind, Span), Error> {
    if let Some(x) = self.rem.first() {
      Ok((&x.kind, x.span))
    } else {
      Err(Error::Unexpected(Span::new(self.last_idx, 0)))
    }
  }
}

trait Parse: Sized {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self>;
}

type PResult<T> = Result<T, Error>;

// # Patterns

#[derive(Clone, Debug, Eq, PartialEq)]
struct RefMutPat {
  pat: Box<Pattern>,
  span: Span,
  span_min: Span,
}

impl RefMutPat {
  fn new(pat: Pattern, span: Span, span_min: Span) -> Self {
    Self { pat: Box::new(pat), span, span_min }
  }
}

impl Display for RefMutPat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "&mut {}", self.pat)
  }
}

impl Parse for RefMutPat {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::AmpMut])?;
    let pat = Pattern::parse(cx)?;
    let span = span1.extend_across(pat.span());
    Ok(Self::new(pat, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RefPat {
  pat: Box<Pattern>,
  span: Span,
  span_min: Span,
}

impl RefPat {
  fn new(pat: Pattern, span: Span, span_min: Span) -> Self {
    Self { pat: Box::new(pat), span, span_min }
  }
}

impl Display for RefPat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match *self.pat {
      Pattern::Binding(ref x) if x.is_mut => {
        write!(f, "&({})", self.pat)
      }
      _ => write!(f, "&{}", self.pat),
    }
  }
}

impl Parse for RefPat {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::Amp])?;
    let pat = Pattern::parse(cx)?;
    let span = span1.extend_across(pat.span());
    Ok(Self::new(pat, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SlicePat {
  pat: Box<Pattern>,
  span: Span,
  span_min: Span,
}

impl SlicePat {
  fn new(pat: Pattern, span: Span, span_min: Span) -> Self {
    Self { pat: Box::new(pat), span, span_min }
  }
}

impl Display for SlicePat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[{}]", self.pat)
  }
}

impl Parse for SlicePat {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::SliceStart])?;
    let pat = Pattern::parse(cx)?;
    let span2 = cx.expect(&[TokenKind::SliceEnd])?;
    let span = span1.extend_across(span2);
    Ok(Self::new(pat, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParenPat {
  pat: Box<Pattern>,
  span: Span,
  span_min: Span,
}

impl ParenPat {
  fn new(pat: Pattern, span: Span, span_min: Span) -> Self {
    Self { pat: Box::new(pat), span, span_min }
  }
}

impl Display for ParenPat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "({})", self.pat)
  }
}

impl Parse for ParenPat {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::ParenStart])?;
    let pat = Pattern::parse(cx)?;
    let span2 = cx.expect(&[TokenKind::ParenEnd])?;
    let span = span1.extend_across(span2);
    Ok(Self::new(pat, span, span1))
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum BindingMode {
  Move,
  RefMut,
  Ref,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BindingPat {
  ident: Ident,
  mode: BindingMode,
  is_mut: bool,
  span: Span,
}

impl BindingPat {
  fn new(
    ident: Ident,
    mode: BindingMode,
    is_mut: bool,
    span: Span,
  ) -> Self {
    Self { ident, mode, is_mut, span }
  }
}

impl Display for BindingPat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use BindingMode::*;
    match (&self.ident, self.mode, self.is_mut) {
      (x, Move, true) => write!(f, "mut {x}"),
      (x, RefMut, false) => write!(f, "ref mut {x}"),
      (x, Ref, false) => write!(f, "ref {x}"),
      (x, _, false) => write!(f, "{x}"),
      _ => unimplemented!(),
    }
  }
}

impl Parse for BindingPat {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let mut mode = BindingMode::Move;
    let mut is_mut = false;
    let mut start_span = None;
    loop {
      let (kind, span) = cx.next_token()?;
      if let None = start_span {
        start_span = Some(span);
      }
      match kind {
        TokenKind::Ref if is_mut => {
          break Err(Error::Unexpected(span))
        }
        TokenKind::Ref => {
          mode = BindingMode::Ref;
          continue;
        }
        TokenKind::RefMut if is_mut => {
          break Err(Error::Unexpected(span));
        }
        TokenKind::RefMut => {
          mode = BindingMode::RefMut;
          continue;
        }
        TokenKind::Mut if !matches!(mode, BindingMode::Move) => {
          break Err(Error::Unexpected(span));
        }
        TokenKind::Mut => {
          is_mut = true;
          continue;
        }
        _ => {}
      }
      let TokenKind::Binding(ref ident) = kind else {
        let span1 = span.with_len(1);
        return Err(Error::Expected(
          span,
          TokenKind::Binding(Ident::new("x".into(), span1)),
        ));
      };
      let ident = ident.clone();
      let span = start_span.unwrap().extend_across(span);
      break Ok(Self::new(ident, mode, is_mut, span));
    }
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Pattern {
  RefMut(RefMutPat),
  Ref(RefPat),
  Slice(SlicePat),
  Paren(ParenPat),
  Binding(BindingPat),
}

impl Pattern {
  fn parse_parens<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    cx.expect(&[TokenKind::ParenStart])?;
    let x = Self::parse(cx)?;
    cx.expect(&[TokenKind::ParenEnd])?;
    Ok(x)
  }

  fn span(&self) -> Span {
    match self {
      Pattern::RefMut(x) => x.span,
      Pattern::Ref(x) => x.span,
      Pattern::Slice(x) => x.span,
      Pattern::Paren(x) => x.span,
      Pattern::Binding(x) => x.span,
    }
  }

  fn span_min(&self) -> Span {
    match self {
      Pattern::RefMut(x) => x.span_min,
      Pattern::Ref(x) => x.span_min,
      Pattern::Slice(x) => x.span_min,
      Pattern::Paren(x) => x.span_min,
      Pattern::Binding(x) => x.span,
    }
  }

  #[must_use]
  fn map<F: Fn(Self) -> Self>(self, f: F) -> Self {
    match f(self) {
      s @ Pattern::Binding(_) => s,
      Pattern::RefMut(mut x) => {
        x.pat = Box::new((*x.pat).map(f));
        Pattern::RefMut(x)
      }
      Pattern::Ref(mut x) => {
        x.pat = Box::new((*x.pat).map(f));
        Pattern::Ref(x)
      }
      Pattern::Slice(mut x) => {
        x.pat = Box::new((*x.pat).map(f));
        Pattern::Slice(x)
      }
      Pattern::Paren(mut x) => {
        x.pat = Box::new((*x.pat).map(f));
        Pattern::Paren(x)
      }
    }
  }

  #[must_use]
  fn simplify(self) -> Self {
    self.map(|pat| match pat {
      Pattern::Binding(_) => pat,
      Pattern::RefMut(_) => pat,
      Pattern::Ref(_) => pat,
      Pattern::Slice(_) => pat,
      Pattern::Paren(x) => *x.pat,
    })
  }
}

impl Display for Pattern {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Pattern::RefMut(x) => write!(f, "{x}"),
      Pattern::Ref(x) => write!(f, "{x}"),
      Pattern::Slice(x) => write!(f, "{x}"),
      Pattern::Paren(x) => write!(f, "{x}"),
      Pattern::Binding(x) => write!(f, "{x}"),
    }
  }
}

impl Parse for Pattern {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    use TokenKind::*;
    Ok(match cx.peek_expect()? {
      (Amp, span) => Pattern::Ref(RefPat::parse(cx)?),
      (AmpMut, span) => Pattern::RefMut(RefMutPat::parse(cx)?),
      (Ref, span) => Pattern::Binding(BindingPat::parse(cx)?),
      (RefMut, span) => Pattern::Binding(BindingPat::parse(cx)?),
      (Mut, span) => Pattern::Binding(BindingPat::parse(cx)?),
      (ParenStart, span) => Pattern::Paren(ParenPat::parse(cx)?),
      (SliceStart, span) => Pattern::Slice(SlicePat::parse(cx)?),
      (Binding(_), span) => Pattern::Binding(BindingPat::parse(cx)?),
      (_, span) => Err(Error::Unexpected(span))?,
    })
  }
}

// # Expressions

#[derive(Clone, Debug, Eq, PartialEq)]
struct TypeExpr {
  name: Ident,
  span: Span,
}

impl TypeExpr {
  fn new(name: Ident, span: Span) -> Self {
    Self { name, span }
  }
}

impl Display for TypeExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.name)
  }
}

impl Parse for TypeExpr {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let (kind, span) = cx.next_token()?;
    let TokenKind::Type(ref ident) = kind else {
      let span1 = span.with_len(1);
      return Err(Error::Expected(
        span,
        TokenKind::Type(Ident::new("T".into(), span1)),
      ));
    };
    Ok(Self::new(ident.clone(), span))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RefMutExpr {
  expr: Box<Expr>,
  span: Span,
  span_min: Span,
}

impl RefMutExpr {
  fn new(expr: Expr, span: Span, span_min: Span) -> Self {
    Self { expr: Box::new(expr), span, span_min }
  }
}

impl Display for RefMutExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "&mut {}", self.expr)
  }
}

impl Parse for RefMutExpr {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::AmpMut])?;
    let expr = Expr::parse(cx)?;
    let span = span1.extend_across(expr.span());
    Ok(Self::new(expr, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RefExpr {
  expr: Box<Expr>,
  span: Span,
  span_min: Span,
}

impl RefExpr {
  fn new(expr: Expr, span: Span, span_min: Span) -> Self {
    Self { expr: Box::new(expr), span, span_min }
  }
}

impl Display for RefExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "&{}", self.expr)
  }
}

impl Parse for RefExpr {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::Amp])?;
    let expr = Expr::parse(cx)?;
    let span = span1.extend_across(expr.span());
    Ok(Self::new(expr, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SliceExpr {
  expr: Box<Expr>,
  span: Span,
  span_min: Span,
}

impl SliceExpr {
  fn new(expr: Expr, span: Span, span_min: Span) -> Self {
    Self { expr: Box::new(expr), span, span_min }
  }
}

impl Display for SliceExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[{}]", self.expr)
  }
}

impl Parse for SliceExpr {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::SliceStart])?;
    let expr = Expr::parse(cx)?;
    let span2 = cx.expect(&[TokenKind::SliceEnd])?;
    let span = span1.extend_across(span2);
    Ok(Self::new(expr, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParenExpr {
  expr: Box<Expr>,
  span: Span,
  span_min: Span,
}

impl ParenExpr {
  fn new(expr: Expr, span: Span, span_min: Span) -> Self {
    Self { expr: Box::new(expr), span, span_min }
  }
}

impl Display for ParenExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "({})", self.expr)
  }
}

impl Parse for ParenExpr {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::ParenStart])?;
    let expr = Expr::parse(cx)?;
    let span2 = cx.expect(&[TokenKind::ParenEnd])?;
    let span = span1.extend_across(span2);
    Ok(Self::new(expr, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Expr {
  Type(TypeExpr),
  RefMut(RefMutExpr),
  Ref(RefExpr),
  Slice(SliceExpr),
  Paren(ParenExpr),
}

impl Expr {
  fn parse_parens<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    cx.expect(&[TokenKind::ParenStart])?;
    let x = Self::parse(cx)?;
    cx.expect(&[TokenKind::ParenEnd])?;
    Ok(x)
  }

  fn span(&self) -> Span {
    match self {
      Expr::Type(x) => x.span,
      Expr::RefMut(x) => x.span,
      Expr::Ref(x) => x.span,
      Expr::Slice(x) => x.span,
      Expr::Paren(x) => x.span,
    }
  }

  fn span_min(&self) -> Span {
    match self {
      Expr::Type(x) => x.span,
      Expr::RefMut(x) => x.span_min,
      Expr::Ref(x) => x.span_min,
      Expr::Slice(x) => x.span_min,
      Expr::Paren(x) => x.span_min,
    }
  }

  #[must_use]
  fn map<F: Fn(Expr) -> Expr>(self, f: F) -> Self {
    match f(self) {
      s @ Expr::Type(_) => s,
      Expr::RefMut(mut x) => {
        x.expr = Box::new((*x.expr).map(f));
        Expr::RefMut(x)
      }
      Expr::Ref(mut x) => {
        x.expr = Box::new((*x.expr).map(f));
        Expr::Ref(x)
      }
      Expr::Slice(mut x) => {
        x.expr = Box::new((*x.expr).map(f));
        Expr::Slice(x)
      }
      Expr::Paren(mut x) => {
        x.expr = Box::new((*x.expr).map(f));
        Expr::Paren(x)
      }
    }
  }

  #[must_use]
  fn simplify(self) -> Self {
    self.map(|expr| match expr {
      Expr::Type(_) => expr,
      Expr::RefMut(_) => expr,
      Expr::Ref(_) => expr,
      Expr::Slice(_) => expr,
      Expr::Paren(x) => *x.expr,
    })
  }
}

impl Display for Expr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Expr::Type(x) => write!(f, "{x}"),
      Expr::RefMut(x) => write!(f, "{x}"),
      Expr::Ref(x) => write!(f, "{x}"),
      Expr::Slice(x) => write!(f, "{x}"),
      Expr::Paren(x) => write!(f, "{x}"),
    }
  }
}

impl Parse for Expr {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    use TokenKind::*;
    Ok(match cx.peek_expect()? {
      (Amp, span) => Expr::Ref(RefExpr::parse(cx)?),
      (AmpMut, span) => Expr::RefMut(RefMutExpr::parse(cx)?),
      (ParenStart, span) => Expr::Paren(ParenExpr::parse(cx)?),
      (SliceStart, _) => Expr::Slice(SliceExpr::parse(cx)?),
      (Type(_), _) => Expr::Type(TypeExpr::parse(cx)?),
      (_, span) => Err(Error::Unexpected(span))?,
    })
  }
}

// # Statements

#[derive(Clone, Debug, Eq, PartialEq)]
struct LetStmt {
  pat: Pattern,
  expr: Expr,
  span: Span,
  span_min: Span,
}

impl LetStmt {
  fn new(
    pat: Pattern,
    expr: Expr,
    span: Span,
    span_min: Span,
  ) -> Self {
    Self { pat, expr, span, span_min }.simplify()
  }

  fn from_str(xs: &str) -> Result<LetStmt, Error> {
    let xs = lex(xs)?;
    let mut cx = Ctx::new(&xs);
    LetStmt::parse(&mut cx).map(|x| x.simplify())
  }

  fn from_pat_expr(pat: &Pattern, expr: &Expr) -> Self {
    let xs = format!("let {} = {};", pat, expr);
    Self::from_str(&xs).unwrap()
  }

  fn as_string(&self) -> String {
    format!("{}", self)
  }

  #[must_use]
  fn simplify(self) -> Self {
    Self {
      pat: self.pat.simplify(),
      expr: self.expr.simplify(),
      span: self.span,
      span_min: self.span_min,
    }
  }

  #[must_use]
  fn reparse(self) -> Self {
    let xs = lex(&format!("{}", self)).unwrap();
    let mut cx = Ctx::new(&xs);
    Self::parse(&mut cx).unwrap().simplify()
  }

  fn parsing_marks(&self) -> ParsingMarks<'_> {
    ParsingMarks(self)
  }
}

struct ParsingMarks<'s>(&'s LetStmt);

impl<'s> Display for ParsingMarks<'s> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let pat_span = self.0.pat.span_min();
    let expr_span = self.0.expr.span_min();
    for _ in 0..pat_span.start {
      write!(f, " ")?;
    }
    for _ in 0..pat_span.len() {
      write!(f, "^")?;
    }
    for _ in 0..pat_span.len_until(expr_span) {
      write!(f, " ")?;
    }
    for _ in 0..expr_span.len() {
      write!(f, "^")?;
    }
    Ok(())
  }
}

impl Display for LetStmt {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "let {} = {};", self.pat, self.expr)
  }
}

impl Parse for LetStmt {
  fn parse<'x>(cx: &mut Ctx<'x>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::Let])?;
    let pat = Pattern::parse(cx)?;
    _ = cx.expect(&[TokenKind::Equals])?;
    let expr = Expr::parse(cx)?;
    let mut span = span1.extend_across(expr.span());
    if let Ok(span2) = cx.expect(&[TokenKind::Semicolon]) {
      span = span.extend_across(span2);
    }
    Ok(LetStmt::new(pat, expr, span, span1))
  }
}

fn format_let_stmt(xs: &str) -> Result<String, Error> {
  Ok(LetStmt::from_str(xs)?.as_string())
}

#[test]
fn test_let_stmt() {
  let xs =
    "let &[&mut [&&mut [&ref mut xxx]]] = &[&mut [&&mut [&T]]];";
  assert_eq!(format_let_stmt(xs).unwrap(), xs);
  let xs = "let &[&mut [&&mut [&(mut xxx)]]] = &[&mut [&&mut [&T]]];";
  assert_eq!(format_let_stmt(xs).unwrap(), xs);
  let xs = "let &[&mut [&&mut [&x]]] = &[&mut [&&mut [&TTT]]];";
  assert_eq!(format_let_stmt(xs).unwrap(), xs);
}

// # Base64

const B64: [u8; 64] =
  *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64idx(x: u8) -> Option<u8> {
  Some(match x {
    b'A'..=b'Z' => x - b'A',
    b'a'..=b'z' => 26 + x - b'a',
    b'0'..=b'9' => 52 + x - b'0',
    b'-' => 62,
    b'_' => 63,
    _ => None?,
  })
}

fn b64encode(mut xs: &[u8]) -> Vec<u8> {
  let mut xs = xs.iter();
  let mut ys = Vec::with_capacity(((xs.len() / 3) + 1) * 4);
  loop {
    let mut v: u32 = 0;
    let mut i: u32 = 0;
    loop {
      let Some(&x) = xs.next() else { break };
      v |= (x as u32) << (24 - i * 8);
      i += 1;
      if i == 3 {
        break;
      }
    }
    if i == 0 {
      break;
    }
    for j in 0..4 {
      match (i, j) {
        (1, 2..) => ys.push(b'='),
        (2, 3) => ys.push(b'='),
        _ => ys.push(B64[(v >> 26) as usize]),
      }
      v <<= 6;
    }
  }
  ys
}

fn b64decode(mut xs: &[u8]) -> Option<Vec<u8>> {
  let mut xs = xs.iter();
  let mut ys = Vec::with_capacity(((xs.len() / 4) + 1) * 3);
  loop {
    let mut v: u32 = 0;
    let mut i: u32 = 0;
    loop {
      let Some(&x) = xs.next() else { break };
      match (x, i) {
        (b'=', 2) => continue,
        (b'=', 3) => break,
        _ => {}
      }
      v |= (b64idx(x)? as u32) << (26 - i * 6);
      i += 1;
      if i == 4 {
        break;
      }
    }
    match i {
      0 => break,
      1 => return None,
      _ => {}
    }
    for j in 0..3 {
      match (i, j) {
        (2, 1) => break,
        (3, 2) => break,
        _ => ys.push((v >> 24) as u8),
      }
      v <<= 8;
    }
  }
  Some(ys)
}

#[test]
fn test_base64() {
  let mut xs = Vec::<u8>::new();
  for i in 0..1024 {
    let ys = b64decode(&b64encode(&xs)).unwrap();
    assert_eq!(ys, xs);
    xs.push((i % 256) as u8);
  }
}

// # Mermaid

use serde_json::json;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
enum MermaidLinkKind {
  View,
  Edit,
  Img(&'static str),
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct Mermaid<C> {
  link_kind: MermaidLinkKind,
  code: C,
}

impl<C> Mermaid<C> {
  fn new(link_kind: MermaidLinkKind, code: C) -> Self {
    Self { link_kind, code }
  }
}

impl<C: Display> Display for Mermaid<C> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write_mermaid_link(f, &self.code, self.link_kind)
  }
}

fn write_mermaid_link<W: Write, C: Display>(
  w: &mut W,
  code: C,
  kind: MermaidLinkKind,
) -> fmt::Result {
  let mut z = flate2::Compress::new_with_window_bits(
    flate2::Compression::best(),
    true,
    15,
  );
  let code = format!("{}", code);
  let link = json!({
    "code": code,
    "mermaid": json!({"theme": "dark"}),
    "autoSync": false,
    "updateDiagram": false,
  });
  let xs = serde_json::to_vec(&link).unwrap();
  let mut ys = Vec::with_capacity(2048);
  assert!(matches!(
    z.compress_vec(&xs, &mut ys, flate2::FlushCompress::Finish)
      .unwrap(),
    flate2::Status::StreamEnd,
  ));
  let xs = b64encode(&ys);
  let xs = core::str::from_utf8(&xs).unwrap();
  let xs = xs.trim_end_matches('=');
  match kind {
    MermaidLinkKind::View => {
      writeln!(w, "https://mermaid.live/view#pako:{}", xs)
    }
    MermaidLinkKind::Edit => {
      writeln!(w, "https://mermaid.live/edit#pako:{}", xs)
    }
    MermaidLinkKind::Img(ty) => {
      writeln!(w, "https://mermaid.ink/img/pako:{}?type={ty}", xs)
    }
  }
}

// # Graphs

#[derive(
  Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
#[repr(u8)]
enum Node {
  Start,
  Error,
  Move,
  MoveBehindRef,
  RefMut,
  RefMutBehindRef,
  Ref,
}

const NODES: &[Node] = {
  use Node::*;
  &[Move, MoveBehindRef, RefMut, RefMutBehindRef, Ref]
};

impl Node {
  fn label(&self) -> &'static str {
    match self {
      Node::Start => "Start",
      Node::Error => "Error",
      Node::Move => "move",
      Node::MoveBehindRef => "move behind &",
      Node::RefMut => "ref mut",
      Node::RefMutBehindRef => "ref mut behind &",
      Node::Ref => "ref",
    }
  }

  fn attr(&self) -> &'static str {
    match self {
      Node::Start => "start",
      Node::Error => "error",
      Node::Move => "move",
      Node::MoveBehindRef => "move_behind_shared_ref",
      Node::RefMut => "ref_mut",
      Node::RefMutBehindRef => "ref_mut_behind_shared_ref",
      Node::Ref => "ref",
    }
  }

  fn id(&self) -> &'static str {
    match self {
      Node::Start => "I",
      Node::Error => "E",
      Node::Move => "M",
      Node::MoveBehindRef => "N",
      Node::RefMut => "X",
      Node::RefMutBehindRef => "Y",
      Node::Ref => "S",
    }
  }
}

impl Display for Node {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}[\"{}\"]", self.id(), self.label())
  }
}

#[derive(
  Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
#[repr(u8)]
enum EdgeDir {
  Forward,
  Backward,
  ToSelf,
  Error,
}

impl EdgeDir {
  fn color(&self) -> &'static str {
    match self {
      EdgeDir::Forward => "green",
      EdgeDir::Backward => "blue",
      EdgeDir::ToSelf => "gray",
      EdgeDir::Error => "orange",
    }
  }
}

#[derive(
  Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
#[repr(u8)]
enum EdgeTy {
  NrpVsRef,
  NrpVsRefMut,
  NrpVsT,
  RefVsRef,
  RefMutVsRefMut,
  RefVsRefMut,
  RefMutVsRef,
  RefVsT,
  RefMutVsT,
  MutTok,
  RefMutTok,
  RefTok,
  Binding,
  Empty,
}

const EDGE_TYS: &[EdgeTy] = {
  use EdgeTy::*;
  &[
    NrpVsRef,
    NrpVsRefMut,
    NrpVsT,
    RefVsRef,
    RefMutVsRefMut,
    RefVsRefMut,
    RefMutVsRef,
    RefVsT,
    RefMutVsT,
    MutTok,
    RefMutTok,
    RefTok,
    Binding,
  ]
};

impl EdgeTy {
  fn label(&self) -> &'static str {
    match self {
      EdgeTy::NrpVsRef => "[]-&",
      EdgeTy::NrpVsRefMut => "[]-&mut",
      EdgeTy::NrpVsT => "[]-T",
      EdgeTy::RefVsRef => "&-&",
      EdgeTy::RefMutVsRefMut => "&mut-&mut",
      EdgeTy::RefVsRefMut => "&-&mut",
      EdgeTy::RefMutVsRef => "&mut-&",
      EdgeTy::RefVsT => "&-T",
      EdgeTy::RefMutVsT => "&mut-T",
      EdgeTy::MutTok => "mut",
      EdgeTy::RefMutTok => "ref mut",
      EdgeTy::RefTok => "ref",
      EdgeTy::Binding => "x",
      EdgeTy::Empty => "",
    }
  }

  fn description(&self) -> &'static str {
    match self {
      EdgeTy::NrpVsRef => {
        "non-reference pattern matches against shared reference"
      }
      EdgeTy::NrpVsRefMut => {
        "non-reference pattern matches against mutable reference"
      }
      EdgeTy::NrpVsT => {
        "non-reference pattern matches against some non-reference type"
      }
      EdgeTy::RefVsRef => {
        "`&` pattern matches against shared reference type"
      },
      EdgeTy::RefMutVsRefMut => {
        "`&mut` pattern matches against mutable reference type"
      }
      EdgeTy::RefVsRefMut => {
        "`&` pattern matches against mutable reference type"
      }
      EdgeTy::RefMutVsRef => {
        "`&mut` pattern matches against shared reference type"
      }
      EdgeTy::RefVsT => {
        "`&` pattern matches against some non-reference type"
      },
      EdgeTy::RefMutVsT => {
        "`&mut` pattern matches against some non-reference type"
      }
      EdgeTy::MutTok => "binding pattern declared `mut`",
      EdgeTy::RefMutTok => "binding pattern declared `ref mut`",
      EdgeTy::RefTok => "binding pattern declared `ref`",
      EdgeTy::Binding => "binding pattern",
      EdgeTy::Empty => "empty edge",
    }
  }
}

impl Display for EdgeTy {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.label())?;
    if f.alternate() {
      write!(f, " ({})", self.description())?;
    }
    Ok(())
  }
}

#[derive(
  Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
struct Edge {
  from: Node,
  ty: EdgeTy,
  to: Node,
}

impl Edge {
  fn new(from: Node, to: Node, ty: EdgeTy) -> Self {
    Self { from, to, ty }
  }

  fn dir(&self) -> EdgeDir {
    if self.to == Node::Error {
      EdgeDir::Error
    } else if self.from == self.to {
      EdgeDir::ToSelf
    } else if self.to > self.from {
      EdgeDir::Forward
    } else {
      EdgeDir::Backward
    }
  }
}

impl Display for Edge {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if self.ty == EdgeTy::Empty {
      write!(f, "{}", self.from.id())?;
      write!(f, "{:14}--> ", "")?;
      write!(f, "{}", self.to.id())
    } else {
      write!(f, "{} -- ", self.from.id())?;
      write!(f, "{:9} --> ", self.ty.label())?;
      write!(f, "{}", self.to.id())
    }
  }
}

#[derive(Debug)]
struct Graph {
  edges: Vec<Edge>,
}

impl Graph {
  fn new() -> Self {
    Self { edges: Vec::new() }
  }

  fn add_edge(&mut self, from: Node, to: Node, ty: EdgeTy) {
    self.edges.push(Edge::new(from, to, ty))
  }

  fn nodes(&self) -> Vec<Node> {
    let mut xs = HashSet::new();
    for x in &self.edges {
      xs.insert(x.from);
      xs.insert(x.to);
    }
    let mut xs = xs.drain().collect::<Vec<_>>();
    xs.sort();
    xs
  }

  fn sort_edges(&mut self) {
    self.edges.sort();
  }

  /// Delete edges from terminal nodes.
  fn simplify_terminal(&mut self) {
    let mut h = HashSet::new();
    for x in &self.edges {
      if x.from != x.to {
        h.insert(x.from);
      }
    }
    self.edges.retain(|x| h.contains(&x.from));
  }

  /// Delete edge types that always recurse.
  fn simplify_always_recurse(&mut self) {
    let mut h = HashSet::new();
    for x in &self.edges {
      if x.from != x.to {
        h.insert(x.ty);
      }
    }
    self.edges.retain(|x| h.contains(&x.ty));
  }

  /// Delete error edges.
  fn remove_error_edges(&mut self) {
    self.edges.retain(|x| x.to != Node::Error);
  }

  /// Delete recursive edges.
  fn remove_recursive_edges(&mut self) {
    self.edges.retain(|x| x.from != x.to);
  }

  /// Delete redundant nodes.
  // A node is redundant if all of its non-self outgoing edges match
  // the outgoing edges of some other node.
  fn remove_redundant_nodes(&mut self) {
    let mut m = HashMap::new();
    for x in &self.edges {
      if x.from != x.to {
        let mut s = m.entry(x.from).or_insert_with(HashSet::new);
        s.insert((x.ty, x.to));
      }
    }
    let mut rm = HashMap::new();
    for node in NODES {
      if let Some(s) = m.get(node) {
        let mut s =
          s.iter().map(|&(x, y)| (x, y)).collect::<Vec<_>>();
        s.sort();
        if let Some(prev_node) = rm.get(&s) {
          for x in &mut self.edges {
            if x.to == *node {
              x.to = *prev_node;
            }
          }
          self.edges.retain(|x| x.from != *node);
        } else {
          rm.insert(s, *node);
        }
      }
    }
  }

  fn simplify_recuse(&mut self) {
    self.simplify_terminal();
    self.simplify_always_recurse();
    self.sort_edges();
  }

  fn simplify_error(&mut self) {
    self.remove_error_edges();
    self.sort_edges();
  }

  fn edge_idxs_by_dir(&self) -> HashMap<EdgeDir, HashSet<usize>> {
    let mut xs = HashMap::new();
    for (i, x) in self.edges.iter().enumerate() {
      let s = xs.entry(x.dir()).or_insert(HashSet::new());
      s.insert(i);
    }
    xs
  }

  fn write_link_styles<W: Write>(&self, w: &mut W) -> fmt::Result {
    let mut v = self.edge_idxs_by_dir();
    let mut vs = Vec::new();
    for (k, v) in v.drain() {
      vs.push((k, v));
    }
    vs.sort_by_key(|x| x.0);
    for x in vs {
      let (dir, mut idxs) = x;
      let mut idxs = idxs.drain().collect::<Vec<_>>();
      idxs.sort();
      write!(w, "linkStyle ")?;
      for (i, x) in idxs.iter().enumerate() {
        write!(w, "{}", x)?;
        if i + 1 != idxs.len() {
          write!(w, ",")?;
        }
      }
      writeln!(w, " stroke:{}", dir.color())?;
    }
    Ok(())
  }

  fn write_header<W: Write>(&self, w: &mut W) -> fmt::Result {
    writeln!(
      w,
      "{}",
      "%%{ init: { 'flowchart': { 'curve': 'linear' } } }%%\n"
    )?;
    writeln!(w, "flowchart LR\n")
  }

  fn write_nodes<W: Write>(&self, w: &mut W) -> fmt::Result {
    for x in self.nodes() {
      writeln!(w, "{}", x)?;
    }
    writeln!(w)
  }

  fn write_edges<W: Write>(
    &self,
    w: &mut W,
    debug: bool,
  ) -> fmt::Result {
    let mut last: Option<Node> = None;
    for (i, x) in self.edges.iter().enumerate() {
      if let Some(l) = last {
        if l != x.from {
          writeln!(w)?;
        }
      }
      last = Some(x.from);
      if debug {
        writeln!(w, "%% edge: {i}")?;
      }
      writeln!(w, "{}", x)?;
    }
    writeln!(w)
  }
}

impl Display for Graph {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.write_header(f)?;
    self.write_nodes(f)?;
    self.write_edges(f, f.alternate())?;
    self.write_link_styles(f)?;
    Ok(())
  }
}

// # Transitions

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
struct Conf {
  /// Disable match ergonomics entirely.
  no_me: bool,
  /// When the default binding mode is not `move`, writing `mut` on a
  /// binding is an error.
  rule1: bool,
  /// When a reference pattern matches against a reference, do not
  /// update the default binding mode.
  rule2: bool,
  /// Keep track of whether we have matched either a reference pattern
  /// or a non-reference pattern against a shared reference, and if
  /// so, set the DBM to `ref` when we would otherwise set it to `ref
  /// mut`.
  rule3: bool,
  /// If a reference pattern is being matched against a non-reference
  /// type and if the DBM is `ref` or `ref mut`, match the pattern
  /// against the DBM as though it were a type.
  rule4: bool,
  /// If the DBM is `ref` or `ref mut`, match a reference pattern
  /// against it as though it were a type *before* considering the
  /// scrutinee.
  rule4_early: bool,
  /// If a `&` pattern is being matched against a type of mutable
  /// reference (or against a `ref mut` DBM under *Rule 4*), act as
  /// though the type were a shared reference instead (or that a `ref
  /// mut` DBM is a `ref` DBM instead).
  rule5: bool,
  /// Rule 3, but lazily applied.
  rule3_lazy: bool,
  /// Spin rule.
  spin: bool,
}

impl Conf {
  fn no_me() -> Self {
    Self { no_me: true, ..<_>::default() }
  }

  fn rust_2021() -> Self {
    Self::default()
  }

  fn rust_2021_proposed() -> Self {
    Self { rule3: true, rule4: true, rule5: true, ..<_>::default() }
  }

  fn rust_2024() -> Self {
    Self { rule1: true, rule2: true, ..<_>::default() }
  }

  fn rust_2024_proposed() -> Self {
    Self {
      rule1: true,
      rule2: true,
      rule3: true,
      rule4: true,
      rule5: true,
      ..<_>::default()
    }
  }

  fn rust_2024_rpjohnst() -> Self {
    Self {
      rule1: true,
      rule2: true,
      rule3_lazy: true,
      rule4: true,
      rule4_early: true,
      spin: true,
      ..<_>::default()
    }
  }
}

#[derive(
  Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd,
)]
#[repr(u8)]
enum StepNote {
  LazilySetToRef,
  EagerlySetToRef,
  SpinHappened,
  LeakToMove,
  MutResetsDBM,
  AppliedRule1,
  AppliedRule2,
  AppliedRule3,
  AppliedRule3Lazy,
  AppliedRule4,
  AppliedRule4Early,
  AppliedRule5,
  AppliedSpin,
}

impl StepNote {
  fn description(&self) -> &'static str {
    match self {
      StepNote::LazilySetToRef => {
        "sets binding to `ref` lazily behind reference"
      }
      StepNote::EagerlySetToRef => {
        "sets DBM to `ref` eagerly behind reference"
      }
      StepNote::SpinHappened => {
        "\"spins\" back from `ref mut` to `ref` DBM"
      }
      StepNote::LeakToMove => {
        "\"leaks\" from `ref` to `move` DBM behind reference"
      }
      StepNote::MutResetsDBM => "`mut` on binding resets the DBM",
      StepNote::AppliedRule1 => {
        "apply Rule 1: error on `mut` with non-move DBM"
      }
      StepNote::AppliedRule2 => {
        "apply Rule 2: preserve DBM on `&-&`/`&mut-&mut` match"
      },
      StepNote::AppliedRule3 => {
        "apply Rule 3: set DBM to `ref` behind `&`"
      }
      StepNote::AppliedRule3Lazy => {
        "apply lazy Rule 3: set DBM to `ref` behind `&` on bindings"
      }
      StepNote::AppliedRule4 => {
        "apply Rule 4: match `&/&mut` against DBM if no structural match"
      }
      StepNote::AppliedRule4Early => {
        "apply early Rule 4: match `&/&mut` against DBM ahead of structural match"
      },
      StepNote::AppliedRule5 => {
        "apply Rule 5: match `&` against `&mut"
      },
      StepNote::AppliedSpin => {
        "apply Spin rule: move from `ref` to `ref mut` DBM"
      },
    }
  }
}

impl Display for StepNote {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.description())
  }
}

#[derive(
  Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd,
)]
#[repr(u8)]
enum StepError {
  MutTokNotAllowedBehindRef,
  RefMutNotAllowedBehindRef,
  MEDisabled,
  AlreadyError,
  TypeMismatch,
}

impl StepError {
  fn description(&self) -> &'static str {
    match self {
      StepError::MutTokNotAllowedBehindRef => {
        "`mut` on a binding is not allowed behind a reference"
      }
      StepError::RefMutNotAllowedBehindRef => {
        "cannot borrow as mutable"
      }
      StepError::MEDisabled => "not allowed without match ergonomics",
      StepError::AlreadyError => "errors stay errors",
      StepError::TypeMismatch => "mismatched types",
    }
  }
}

impl Display for StepError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.description())
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NodeStep {
  conf: Conf,
  node: Node,
  behind_ref: bool,
  notes: HashSet<StepNote>,
  error: Option<StepError>,
  last: bool,
  skip_expr: bool,
  skip_pat: bool,
}

impl NodeStep {
  fn new(conf: Conf, node: Node) -> Self {
    Self {
      conf,
      node,
      behind_ref: false,
      notes: HashSet::new(),
      error: None,
      last: false,
      skip_expr: false,
      skip_pat: false,
    }
  }

  fn add_note(&mut self, note: StepNote) {
    self.notes.insert(note);
  }

  fn node(&mut self, node: Node, note: StepNote) -> Self {
    assert!(!matches!(node, Node::Error));
    self.add_note(note);
    self.node = node;
    self.clone()
  }

  fn node_last(&mut self, node: Node, note: StepNote) -> Self {
    assert!(!matches!(node, Node::Error));
    self.add_note(note);
    self.node = node;
    self.last = true;
    self.clone()
  }

  fn err(&mut self, err: StepError) -> Self {
    self.error = Some(err);
    self.node = Node::Error;
    self.last = true;
    self.clone()
  }

  fn node1(&mut self, node: Node) -> Self {
    assert!(!matches!(node, Node::Error));
    self.node = node;
    self.clone()
  }

  fn node1_last(&mut self, node: Node) -> Self {
    assert!(!matches!(node, Node::Error));
    self.node = node;
    self.last = true;
    self.clone()
  }

  fn notes(&self) -> Vec<StepNote> {
    let mut xs = self.notes.iter().map(|x| *x).collect::<Vec<_>>();
    xs.sort();
    xs
  }
}

fn node_step(
  node: Node,
  mut edge_ty: EdgeTy,
  conf: Conf,
) -> NodeStep {
  use EdgeTy::*;
  use Node::*;
  use StepError::*;
  use StepNote::*;
  let Conf {
    no_me,
    rule1,
    rule2,
    rule3,
    rule4,
    rule4_early,
    rule5,
    rule3_lazy,
    spin,
  } = conf;
  let mut ns = NodeStep::new(conf, node);
  match (ns.node, edge_ty) {
    (MoveBehindRef, _) => {
      ns.behind_ref = true;
      ns.node = Move;
    }
    (RefMutBehindRef, _) if rule3 => panic!(),
    (RefMutBehindRef, _) => {
      ns.behind_ref = true;
      ns.node = RefMut;
    }
    (Ref, _) => ns.behind_ref = true,
    (_, NrpVsRef | RefVsRef | RefVsRefMut | RefMutVsRef | RefVsT) => {
      ns.behind_ref = true;
    }
    _ => {}
  }
  match edge_ty {
    MutTok | RefMutTok | RefTok | Binding => ns.skip_expr = true,
    NrpVsRef | NrpVsRefMut => ns.skip_pat = true,
    _ => {}
  }
  if no_me {
    match (ns.node, edge_ty) {
      (_, Binding) => {}
      (RefMut | Ref, _) => return ns.err(MEDisabled),
      (_, NrpVsRef | NrpVsRefMut) => return ns.err(MEDisabled),
      _ => {}
    }
  }
  if rule1 {
    match (ns.node, edge_ty) {
      (Move, _) => {}
      (_, MutTok) => {
        ns.add_note(AppliedRule1);
        return ns.err(MutTokNotAllowedBehindRef);
      }
      _ => {}
    }
  }
  if rule5 {
    match (ns.node, edge_ty) {
      (_, RefVsRefMut) => {
        ns.add_note(AppliedRule5);
        edge_ty = RefVsRef;
      }
      (RefMut, RefVsT) => {
        ns.add_note(AppliedRule5);
        ns.node = Ref;
      }
      _ => {}
    }
  }
  let mut next_node = match (ns.node, edge_ty) {
    (Start, _) => Some(ns.node1(Move)),
    (_, Empty) => panic!(),
    _ => None,
  };
  if rule2 {
    next_node = match (ns.node, edge_ty) {
      (n @ (RefMut | Ref), RefVsRef | RefMutVsRefMut) => {
        ns.add_note(AppliedRule2);
        Some(ns.node1(n))
      }
      _ => next_node,
    };
  }
  if rule4_early {
    next_node = match (ns.node, edge_ty) {
      (RefMut, RefMutVsRefMut) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4Early);
        Some(ns.node1(Move))
      }
      (Ref, RefVsRef) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4Early);
        Some(ns.node1(Move))
      }
      (Ref, RefMutVsRefMut) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4Early);
        Some(ns.err(TypeMismatch))
      }
      _ => next_node,
    };
  }
  if rule4 {
    next_node = match (ns.node, edge_ty) {
      (RefMut, RefMutVsT) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4);
        Some(ns.node1(Move))
      }
      (Ref, RefVsT) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4);
        Some(ns.node1(Move))
      }
      _ => next_node,
    };
  }
  let next_node = if let Some(next_node) = next_node {
    next_node
  } else {
    match (ns.node, edge_ty) {
      (Start | MoveBehindRef | RefMutBehindRef, _) => unreachable!(),
      (_, Empty) => unreachable!(),
      (Error, _) => ns.err(AlreadyError),
      (node, NrpVsT) => ns.node1(node),
      (_, RefVsRefMut | RefMutVsRef | RefVsT | RefMutVsT) => {
        ns.err(TypeMismatch)
      }
      (_, NrpVsRef) => ns.node1(Ref),
      (Move | RefMut, NrpVsRefMut) => ns.node1(RefMut),
      (Ref, NrpVsRefMut) if spin => {
        ns.add_note(AppliedSpin);
        ns.node(RefMut, SpinHappened)
      }
      (Ref, NrpVsRefMut) => ns.node(Ref, EagerlySetToRef),
      (Ref, RefVsRef | RefMutVsRefMut) if ns.behind_ref && !rule3 => {
        ns.node(Move, LeakToMove)
      }
      (_, RefVsRef | RefMutVsRefMut) => ns.node1(Move),
      (RefMut | Ref, MutTok) => ns.node(Move, MutResetsDBM),
      (_, MutTok) => ns.node1(Move),
      (Move | RefMut, RefMutTok) => ns.node1(RefMut),
      (Ref, RefMutTok) => ns.err(RefMutNotAllowedBehindRef),
      (_, RefTok) => ns.node1(Ref),
      (node, Binding) => ns.node1_last(node),
    }
  };
  if ns.behind_ref {
    match (next_node.node, edge_ty) {
      (_, RefMutTok) => ns.err(RefMutNotAllowedBehindRef),
      (Move, _) => ns.node1(MoveBehindRef),
      (RefMut, Binding) if rule3_lazy => {
        ns.add_note(AppliedRule3Lazy);
        ns.node_last(Ref, LazilySetToRef)
      }
      (RefMut, _) if rule3 => {
        ns.add_note(AppliedRule3);
        ns.node(Ref, EagerlySetToRef)
      }
      (RefMut, Binding) => ns.err(RefMutNotAllowedBehindRef),
      (RefMut, _) => ns.node1(RefMutBehindRef),
      _ => next_node,
    }
  } else {
    next_node
  }
}

fn walk_graph<F: FnMut(Node, EdgeTy, Node)>(
  node: Node,
  conf: Conf,
  hs: &mut HashSet<(Node, EdgeTy, Node)>,
  f: &mut F,
) {
  for edge_ty in EDGE_TYS {
    let next_node = node_step(node, *edge_ty, conf).node;
    if hs.insert((node, *edge_ty, next_node)) {
      f(node, *edge_ty, next_node);
      walk_graph(next_node, conf, hs, f);
    }
  }
}

fn make_graph(conf: Conf) -> Graph {
  use EdgeTy::*;
  use Node::*;
  let mut g = Graph::new();
  g.add_edge(Start, Move, Empty);
  let mut hs = HashSet::new();
  walk_graph(Move, conf, &mut hs, &mut |node, edge_ty, next_node| {
    g.add_edge(node, next_node, edge_ty);
  });
  g
}

// # Reduction

fn edge_ty(pat: &Pattern, expr: &Expr) -> EdgeTy {
  use EdgeTy::*;
  match (pat, expr) {
    (_, Expr::Paren(_)) => unimplemented!(),
    (Pattern::Paren(_), _) => unimplemented!(),
    (Pattern::RefMut(_), Expr::Type(_)) => RefMutVsT,
    (Pattern::RefMut(_), Expr::RefMut(_)) => RefMutVsRefMut,
    (Pattern::RefMut(_), Expr::Ref(_)) => RefMutVsRef,
    (Pattern::RefMut(_), Expr::Slice(_)) => RefMutVsT,
    (Pattern::Ref(_), Expr::Type(_)) => RefVsT,
    (Pattern::Ref(_), Expr::RefMut(_)) => RefVsRefMut,
    (Pattern::Ref(_), Expr::Ref(_)) => RefVsRef,
    (Pattern::Ref(_), Expr::Slice(_)) => RefVsT,
    (Pattern::Slice(_), Expr::Type(_)) => NrpVsT,
    (Pattern::Slice(_), Expr::RefMut(_)) => NrpVsRefMut,
    (Pattern::Slice(_), Expr::Ref(_)) => NrpVsRef,
    (Pattern::Slice(_), Expr::Slice(_)) => NrpVsT,
    (Pattern::Binding(BindingPat { is_mut: true, .. }), _) => MutTok,
    (Pattern::Binding(BindingPat { mode, .. }), _) => match mode {
      BindingMode::Move => Binding,
      BindingMode::RefMut => RefMutTok,
      BindingMode::Ref => RefTok,
    },
  }
}

impl Pattern {
  fn reduce_one(&mut self) -> bool {
    match self {
      Pattern::RefMut(ref mut x) => *self = *x.pat.clone(),
      Pattern::Ref(ref mut x) => *self = *x.pat.clone(),
      Pattern::Slice(ref mut x) => *self = *x.pat.clone(),
      Pattern::Paren(_) => unimplemented!(),
      Pattern::Binding(ref mut x) => {
        if x.is_mut {
          x.is_mut = false;
        } else if let BindingMode::RefMut = x.mode {
          x.mode = BindingMode::Move;
        } else if let BindingMode::Ref = x.mode {
          x.mode = BindingMode::Move;
        } else {
          return true;
        }
      }
    }
    false
  }
}

impl Expr {
  fn reduce_one(&mut self) -> bool {
    match self {
      Expr::Type(_) => return true,
      Expr::RefMut(ref mut x) => *self = *x.expr.clone(),
      Expr::Ref(ref mut x) => *self = *x.expr.clone(),
      Expr::Slice(ref mut x) => *self = *x.expr.clone(),
      Expr::Paren(_) => unimplemented!(),
    }
    false
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Reduction {
  conf: Conf,
  pat: Pattern,
  expr: Expr,
  node_step: NodeStep,
  edge_ty: EdgeTy,
  last: bool,
  dbm_applied: bool,
}

impl Reduction {
  fn new(conf: Conf, pat: Pattern, expr: Expr) -> Self {
    let node_step = NodeStep::new(conf, Node::Move);
    let pat = pat.simplify();
    let expr = expr.simplify();
    let edge_ty = edge_ty(&pat, &expr);
    Self {
      conf,
      pat,
      expr,
      node_step,
      edge_ty,
      last: false,
      dbm_applied: false,
    }
  }

  fn from_stmt(conf: Conf, stmt: LetStmt) -> Self {
    Self::new(conf, stmt.pat, stmt.expr)
  }

  fn from_str(conf: Conf, xs: &str) -> Result<Self, Error> {
    Ok(Self::from_stmt(conf, LetStmt::from_str(xs)?))
  }

  fn step(&mut self) {
    assert!(!self.last);
    assert!(!matches!(
      self.node_step.node,
      Node::Start | Node::Error
    ));
    self.node_step =
      node_step(self.node_step.node, self.edge_ty, self.conf);
    self.last |= self.node_step.last;
    if !self.node_step.skip_pat {
      self.last |= self.pat.reduce_one();
    }
    if !self.node_step.skip_expr {
      let last_expr = self.expr.reduce_one();
      if last_expr && !self.last {
        self.node_step = self.node_step.err(StepError::TypeMismatch);
      }
      self.last |= last_expr;
    }
    self.edge_ty = edge_ty(&self.pat, &self.expr);
  }

  fn is_err(&self) -> bool {
    let x = matches!(self.node_step.node, Node::Error);
    let y = self.node_step.error.is_some();
    if x || y {
      assert!(x && y);
    }
    x
  }

  fn as_binding_mode(&self) -> BindingMode {
    match self.node_step.node {
      Node::Start | Node::Error => panic!(),
      Node::Move => BindingMode::Move,
      Node::MoveBehindRef => BindingMode::Move,
      Node::RefMut => BindingMode::RefMut,
      Node::RefMutBehindRef => BindingMode::RefMut,
      Node::Ref => BindingMode::Ref,
    }
  }

  fn apply_dbm(&mut self) {
    let bm = self.as_binding_mode();
    self.pat = self.pat.clone().map(|pat| match pat {
      Pattern::RefMut(_) => pat,
      Pattern::Ref(_) => pat,
      Pattern::Slice(_) => pat,
      Pattern::Paren(_) => pat,
      Pattern::Binding(ref x) if !self.conf.rule1 && x.is_mut => pat,
      Pattern::Binding(ref x @ BindingPat { mode, .. }) => match mode
      {
        BindingMode::Move => {
          let mut x = x.clone();
          x.mode = bm;
          Pattern::Binding(x)
        }
        BindingMode::RefMut => pat,
        BindingMode::Ref => pat,
      },
    });
    self.dbm_applied = true;
  }

  fn as_stmt(&self) -> LetStmt {
    LetStmt::from_pat_expr(&self.pat, &self.expr)
  }

  fn as_type(&self) -> (Ident, Expr) {
    assert!(self.last && !self.is_err());
    let expr = match self.as_binding_mode() {
      BindingMode::Move => self.expr.clone(),
      BindingMode::RefMut => Expr::RefMut(RefMutExpr::new(
        self.expr.clone(),
        Span::nop(),
        Span::nop(),
      )),
      BindingMode::Ref => Expr::Ref(RefExpr::new(
        self.expr.clone(),
        Span::nop(),
        Span::nop(),
      )),
    };
    let ident = match self.pat {
      Pattern::Binding(BindingPat { ref ident, .. }) => ident.clone(),
      _ => Ident::new("_x".into(), Span::nop()),
    };
    (ident, expr)
  }

  fn to_type(&self) -> Result<(Ident, Expr), StepError> {
    let mut r = self.clone();
    while !r.last {
      r.step();
    }
    if let Some(err) = r.node_step.error {
      return Err(err);
    }
    Ok(r.as_type())
  }
}

impl Display for Reduction {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let stmt = self.as_stmt();
    if !self.dbm_applied {
      writeln!(f, "#[dbm({})]", self.node_step.node.attr())?;
    }
    write!(f, "{}", stmt)?;
    if self.last && !self.is_err() {
      let (ident, expr) = self.as_type();
      writeln!(f, " //~ {}: {}", ident, expr)?;
    } else {
      writeln!(f);
    }
    writeln!(f, "{}", stmt.parsing_marks())?;
    let mut arrow = '^';
    if self.dbm_applied {
      writeln!(f, "//~{arrow} NOTE DBM applied")?;
      arrow = '|';
    }
    if !self.last {
      let mut r = self.clone();
      r.step();
      if matches!(r.node_step.node, Node::Error) {
        writeln!(f, "//~{arrow} ERROR")?;
        arrow = '|';
      }
      writeln!(f, "//~{arrow} NOTE matches {:#}", self.edge_ty)?;
      arrow = '|';
      for x in r.node_step.notes() {
        writeln!(f, "//~{arrow} NOTE {}", x)?;
      }
    } else {
      writeln!(f, "//~{arrow} NOTE matches {:#}", self.edge_ty)?;
    }
    writeln!(f);
    Ok(())
  }
}

struct ShowType(LetStmt);

impl ShowType {
  fn from_stmt(x: LetStmt) -> Self {
    Self(x)
  }

  fn from_str(xs: &str) -> Result<Self, Error> {
    Ok(Self(LetStmt::from_str(xs)?))
  }

  fn show_for<W: Write>(
    &self,
    f: &mut W,
    conf: Conf,
    label: &'static str,
  ) -> fmt::Result {
    let mut r = Reduction::from_stmt(conf, self.0.clone());
    write!(f, "//~| {}: ", label)?;
    match r.to_type() {
      Ok((ident, ty)) => writeln!(f, "{ident}: {ty}")?,
      Err(e) => writeln!(f, "ERROR {}", e)?,
    }
    Ok(())
  }
}

impl Display for ShowType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if f.alternate() {
      writeln!(f, "{}", self.0);
    }
    let mut conf = Conf::default();
    self.show_for(f, conf, "* Rust stable")?;
    conf.rule1 = true;
    self.show_for(f, conf, "* Rule 1")?;
    conf.rule2 = true;
    self.show_for(f, conf, "* Rule 2")?;
    writeln!(f, r"//~| |\");
    conf.rule3 = true;
    self.show_for(f, conf, "| * Rule 3")?;
    conf.rule4 = true;
    self.show_for(f, conf, "| * Rule 4")?;
    writeln!(f, r"//~| | |\");
    conf.rule4_early = true;
    self.show_for(f, conf, "| | * Rule 4 (early)")?;
    conf.rule4_early = false;
    conf.rule5 = true;
    self.show_for(f, conf, "| * Rule 5")?;
    let conf = Conf::rust_2024_rpjohnst();
    self.show_for(f, conf, "* Rule4 + Rule4 (early) + rpjohnst")?;
    Ok(())
  }
}

// # Main

use rustyline::{error::ReadlineError, DefaultEditor};

fn help() {
  print!(
    r###"
# Match ergonomics formality

This tool has three main features:

- For a given (simplified) let statement, show the resulting type of
binding under a number of different proposed rulesets.

- For a given set of rules, explain the operation of match ergonomics
step by step.

- For a given set of rules, generate a state transition diagram
showing how the default binding mode (DBM) is allowed to transition
from one state to another based on the pattern and the scrutinee.

To start, type a simple let statement, e.g.:

> let [[x]] = &[&mut [T]];

The tool assumes that all lowercase letters are binding names and all
uppercase letters are unit types (that implement `Copy`).  It supports
only slices, and it does not support multiple bindings (or commas
generally).

When you do this, we'll print out the type of the resulting binding
according to a number of different rulesets.

If you want to see this explained, step by step, according to the
current stable rules, type:

> explain

Then just press return to proceed through the steps.

If you want to see things explained under a different set of rules,
you can `set` and `unset` rules.

The orthogonal rules are:

  - rule1
  - rule2
  - rule3
  - rule4
  - rule5
  - rule3_lazy
  - rule4_early
  - spin_rule

And we provide these aliases:

  - stable: Unset all rules.
  - proposed: Set rules 1-5.
  - rfc: Set rules 1-5 + rule4_early.
  - rpjohnst: Set:
    - rule1
    - rule2
    - rule3_lazy
    - rule4
    - rule4_early
    - spin

So we can write, e.g.:

> set proposed
> unset rule5

...if we want to explore how things work under the proposal without
Rule 5.

Once we've set the rules to our liking, we can see a graph of the
state transition diagram with:

> graph

This will print a link to a mermaid diagram.

To see your current settings, including all rules applied, type:

> show

Press ctrl-d to exit.
"###
  );
}

fn main() {
  let mut rl = DefaultEditor::new().unwrap();
  let mut conf = Conf::rust_2021();
  let mut editable_graph = false;
  let mut remove_error_edges = false;
  let mut remove_always_recursive_edges = false;
  let mut show_graph_source = false;
  let mut cur_stmt = None;
  let mut cur_reduction = None;
  let mut cur_reduction_done = false;
  help();
  loop {
    match rl.readline(">> ") {
      Ok(line) if line.trim() == "help" => {
        rl.add_history_entry(line.as_str());
        help();
      }
      Ok(line) if line.trim_start().starts_with("let ") => {
        rl.add_history_entry(line.as_str());
        let stmt = match LetStmt::from_str(&line) {
          Ok(x) => x,
          Err(err) => {
            print_err(&line, err);
            continue;
          }
        };
        cur_stmt = Some(stmt.clone());
        print!("{:#}", ShowType::from_stmt(stmt));
      }
      Ok(line) if line.trim() == "explain" => {
        let Some(ref stmt) = cur_stmt else {
          println!("ERROR: Provide a statement first.");
          continue;
        };
        let mut r = Reduction::from_stmt(conf, stmt.clone());
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
          if !(line.trim() == "") {
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
        } else {
          if !(line.trim() == "") {
            println!("# Nothing more to show.");
          }
        }
      }
      Ok(line) if line.trim() == "show" => {
        rl.add_history_entry(line.as_str());
        println!("Settings:");
        println!("| rule1: {}", conf.rule1);
        println!("| rule2: {}", conf.rule2);
        println!("| rule3: {}", conf.rule3);
        println!("| rule3_lazy: {}", conf.rule3_lazy);
        println!("| rule4: {}", conf.rule4);
        println!("| rule4_early: {}", conf.rule4_early);
        println!("| rule5: {}", conf.rule5);
        println!("| spin_rule: {}", conf.spin);
        println!("| no_match_ergonomics: {}", conf.no_me);
        println!("| editable_graph: {}", editable_graph);
        println!("| remove_error_edges: {}", remove_error_edges);
        println!(
          "| remove_always_recursive_edges: {}",
          remove_always_recursive_edges
        );
        println!("| show_graph_source: {}", show_graph_source);
      }
      Ok(line) if line.trim() == "graph" => {
        rl.add_history_entry(line.as_str());
        let mut g = make_graph(conf);
        g.simplify_terminal();
        if remove_error_edges {
          g.simplify_error();
        }
        if remove_always_recursive_edges {
          g.simplify_recuse();
        }
        if show_graph_source {
          println!("{}", g);
        }
        if editable_graph {
          print!("{}", Mermaid::new(MermaidLinkKind::Edit, &g));
        } else {
          print!("{}", Mermaid::new(MermaidLinkKind::View, &g));
        }
        println!();
      }
      Ok(line) if line.trim() == "graph svg" => {
        rl.add_history_entry(line.as_str());
        let mut g = make_graph(conf);
        g.simplify_terminal();
        if remove_error_edges {
          g.simplify_error();
        }
        if remove_always_recursive_edges {
          g.simplify_recuse();
        }
        if show_graph_source {
          println!("{}", g);
        }
        println!("{}", Mermaid::new(MermaidLinkKind::Img("svg"), &g));
      }
      Ok(line) if line.trim() == "set editable_graph" => {
        rl.add_history_entry(line.as_str());
        editable_graph = true;
      }
      Ok(line) if line.trim() == "unset editable_graph" => {
        rl.add_history_entry(line.as_str());
        editable_graph = false;
      }
      Ok(line) if line.trim() == "set remove_error_edges" => {
        rl.add_history_entry(line.as_str());
        remove_error_edges = true;
      }
      Ok(line) if line.trim() == "unset remove_error_edges" => {
        rl.add_history_entry(line.as_str());
        remove_error_edges = false;
      }
      Ok(line)
        if line.trim() == "set remove_always_recursive_edges" =>
      {
        rl.add_history_entry(line.as_str());
        remove_always_recursive_edges = true;
      }
      Ok(line)
        if line.trim() == "unset remove_always_recursive_edges" =>
      {
        rl.add_history_entry(line.as_str());
        remove_always_recursive_edges = false;
      }
      Ok(line) if line.trim() == "set show_graph_source" => {
        rl.add_history_entry(line.as_str());
        show_graph_source = true;
      }
      Ok(line) if line.trim() == "unset show_graph_source" => {
        rl.add_history_entry(line.as_str());
        show_graph_source = false;
      }
      Ok(line) if line.trim() == "set stable" => {
        rl.add_history_entry(line.as_str());
        conf = Conf::rust_2021();
      }
      Ok(line) if line.trim() == "set proposed" => {
        rl.add_history_entry(line.as_str());
        conf = Conf::rust_2024_proposed();
      }
      Ok(line) if line.trim() == "set rfc" => {
        rl.add_history_entry(line.as_str());
        conf = Conf::rust_2024_proposed();
        conf.rule4_early = true;
      }
      Ok(line) if line.trim() == "set rpjohnst" => {
        rl.add_history_entry(line.as_str());
        conf = Conf::rust_2024_rpjohnst();
      }
      Ok(line) if line.trim() == "set rule1" => {
        rl.add_history_entry(line.as_str());
        conf.rule1 = true;
      }
      Ok(line) if line.trim() == "unset rule1" => {
        rl.add_history_entry(line.as_str());
        conf.rule1 = false;
      }
      Ok(line) if line.trim() == "set rule2" => {
        rl.add_history_entry(line.as_str());
        conf.rule2 = true;
      }
      Ok(line) if line.trim() == "unset rule2" => {
        rl.add_history_entry(line.as_str());
        conf.rule2 = false;
      }
      Ok(line) if line.trim() == "set rule3" => {
        rl.add_history_entry(line.as_str());
        conf.rule3 = true;
      }
      Ok(line) if line.trim() == "unset rule3" => {
        rl.add_history_entry(line.as_str());
        conf.rule3 = false;
      }
      Ok(line) if line.trim() == "set rule3_lazy" => {
        rl.add_history_entry(line.as_str());
        conf.rule3_lazy = true;
      }
      Ok(line) if line.trim() == "unset rule3_lazy" => {
        rl.add_history_entry(line.as_str());
        conf.rule3_lazy = false;
      }
      Ok(line) if line.trim() == "set rule4" => {
        rl.add_history_entry(line.as_str());
        conf.rule4 = true;
      }
      Ok(line) if line.trim() == "unset rule4" => {
        rl.add_history_entry(line.as_str());
        conf.rule4 = false;
      }
      Ok(line) if line.trim() == "set rule4_early" => {
        rl.add_history_entry(line.as_str());
        conf.rule4_early = true;
      }
      Ok(line) if line.trim() == "unset rule4_early" => {
        rl.add_history_entry(line.as_str());
        conf.rule4_early = false;
      }
      Ok(line) if line.trim() == "set rule5" => {
        rl.add_history_entry(line.as_str());
        conf.rule5 = true;
      }
      Ok(line) if line.trim() == "unset rule5" => {
        rl.add_history_entry(line.as_str());
        conf.rule5 = false;
      }
      Ok(line) if line.trim() == "set spin_rule" => {
        rl.add_history_entry(line.as_str());
        conf.spin = true;
      }
      Ok(line) if line.trim() == "unset spin_rule" => {
        rl.add_history_entry(line.as_str());
        conf.spin = false;
      }
      Ok(line) if line.trim() == "set no_match_ergonomics" => {
        rl.add_history_entry(line.as_str());
        conf.no_me = true;
      }
      Ok(line) if line.trim() == "unset no_match_ergonomics" => {
        rl.add_history_entry(line.as_str());
        conf.no_me = false;
      }
      Ok(line) => {
        rl.add_history_entry(line.as_str());
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
