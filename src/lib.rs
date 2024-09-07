use core::fmt::{self, Display, Write};
use std::collections::{HashMap, HashSet};

// # Extension traits

pub trait StrExt {
  fn strip_prefix2(&self, prefix: &str) -> Option<(usize, &str)>;
}

impl StrExt for str {
  fn strip_prefix2(&self, prefix: &str) -> Option<(usize, &str)> {
    self.starts_with(prefix).then(|| {
      let len = prefix.len();
      (len, &self[len..])
    })
  }
}

// # Shared

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Span {
  pub start: usize,
  pub end: usize,
}

impl Span {
  pub fn new(start: usize, len: usize) -> Self {
    Self { start, end: start + len }
  }
  pub fn nop() -> Self {
    Self { start: 0, end: 0 }
  }
  pub fn from2(span1: Span, span2: Span) -> Self {
    Self { start: span1.start, end: span2.end }
  }
  pub fn len(&self) -> usize {
    self.end - self.start
  }
  pub fn extend(&self, end: usize) -> Self {
    Self { start: self.start, end }
  }
  #[allow(dead_code)]
  pub fn extend_until(&self, span: Span) -> Self {
    Self { start: self.start, end: span.start }
  }
  pub fn extend_across(&self, span: Span) -> Self {
    Self { start: self.start, end: span.end }
  }
  pub fn with_len(&self, len: usize) -> Self {
    Self { start: self.start, end: self.start + len }
  }
  pub fn len_until(&self, span: Span) -> usize {
    assert!(span.start >= self.end);
    span.start - self.end
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ident {
  pub name: String,
  pub span: Span,
}

impl Ident {
  pub fn new(name: String, span: Span) -> Self {
    Self { name, span }
  }
  pub fn len(&self) -> usize {
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
pub enum Error {
  Unexpected(Span),
  Expected(Span, TokenKind),
}

pub fn print_err(xs: &str, err: Error) {
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
  }
}

#[derive(Debug)]
pub enum ConfError {
  UnknownFlag(String),
}

impl Display for ConfError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      ConfError::UnknownFlag(flag) => {
        write!(f, "unknown flag {flag:?}")?;
      }
    }
    Ok(())
  }
}

// # Lexing

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Token {
  pub kind: TokenKind,
  pub span: Span,
}

impl Token {
  pub fn new(kind: TokenKind, idx: usize) -> Self {
    let len = kind.len();
    Self { kind, span: Span::new(idx, len) }
  }
  pub fn new_span(kind: TokenKind, span: Span) -> Self {
    Self { kind, span }
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenKind {
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
  pub fn len(&self) -> usize {
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

pub fn lex(mut xs: &str) -> Result<Vec<Token>, Error> {
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

#[cfg(test)]
pub fn unlex(xs: Vec<Token>) -> String {
  let mut ys = String::new();
  for x in xs {
    use TokenKind::*;
    match x.kind {
      Amp => ys.push('&'),
      AmpMut => ys.push_str("&mut "),
      Ref => ys.push_str("ref "),
      RefMut => ys.push_str("ref mut "),
      Mut => ys.push_str("mut "),
      ParenEnd => ys.push(')'),
      ParenStart => ys.push('('),
      SliceEnd => ys.push(']'),
      SliceStart => ys.push('['),
      Equals => ys.push_str(" = "),
      Let => ys.push_str("let "),
      Semicolon => ys.push(';'),
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
pub struct Ctx<'x> {
  pub rem: &'x [Token],
  pub last_idx: usize,
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
  pub fn new(rem: &'x [Token]) -> Self {
    let mut last_idx = 0;
    if let Some(Token { span, .. }) = rem.first() {
      last_idx = span.start;
    }
    Ctx { rem, last_idx }
  }

  pub fn rem(&mut self, rem: &'x [Token]) {
    if let Some(Token { span, .. }) = rem.first() {
      self.last_idx = span.end;
    }
    self.rem = rem;
  }

  pub fn next_token(&mut self) -> Result<(&TokenKind, Span), Error> {
    if let Some(x) = self.rem.first() {
      self.rem(&self.rem[1..]);
      Ok((&x.kind, x.span))
    } else {
      Err(Error::Unexpected(Span::new(self.last_idx, 0)))
    }
  }

  pub fn expect(&mut self, x: &[TokenKind]) -> Result<Span, Error> {
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

  pub fn peek_expect(&self) -> Result<(&TokenKind, Span), Error> {
    if let Some(x) = self.rem.first() {
      Ok((&x.kind, x.span))
    } else {
      Err(Error::Unexpected(Span::new(self.last_idx, 0)))
    }
  }
}

pub trait Parse: Sized {
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self>;
}

pub type PResult<T> = Result<T, Error>;

// # Patterns

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefMutPat {
  pub pat: Box<Pattern>,
  pub span: Span,
  pub span_min: Span,
}

impl RefMutPat {
  pub fn new(pat: Pattern, span: Span, span_min: Span) -> Self {
    Self { pat: Box::new(pat), span, span_min }
  }
}

impl Display for RefMutPat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "&mut {}", self.pat)
  }
}

impl Parse for RefMutPat {
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::AmpMut])?;
    let pat = Pattern::parse(cx)?;
    let span = span1.extend_across(pat.span());
    Ok(Self::new(pat, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefPat {
  pub pat: Box<Pattern>,
  pub span: Span,
  pub span_min: Span,
}

impl RefPat {
  pub fn new(pat: Pattern, span: Span, span_min: Span) -> Self {
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
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::Amp])?;
    let pat = Pattern::parse(cx)?;
    let span = span1.extend_across(pat.span());
    Ok(Self::new(pat, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlicePat {
  pub pat: Box<Pattern>,
  pub span: Span,
  pub span_min: Span,
}

impl SlicePat {
  pub fn new(pat: Pattern, span: Span, span_min: Span) -> Self {
    Self { pat: Box::new(pat), span, span_min }
  }
}

impl Display for SlicePat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[{}]", self.pat)
  }
}

impl Parse for SlicePat {
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::SliceStart])?;
    let pat = Pattern::parse(cx)?;
    let span2 = cx.expect(&[TokenKind::SliceEnd])?;
    let span = span1.extend_across(span2);
    Ok(Self::new(pat, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParenPat {
  pub pat: Box<Pattern>,
  pub span: Span,
  pub span_min: Span,
}

impl ParenPat {
  pub fn new(pat: Pattern, span: Span, span_min: Span) -> Self {
    Self { pat: Box::new(pat), span, span_min }
  }
}

impl Display for ParenPat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "({})", self.pat)
  }
}

impl Parse for ParenPat {
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::ParenStart])?;
    let pat = Pattern::parse(cx)?;
    let span2 = cx.expect(&[TokenKind::ParenEnd])?;
    let span = span1.extend_across(span2);
    Ok(Self::new(pat, span, span1))
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum BindingMode {
  Move,
  RefMut,
  Ref,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingPat {
  pub ident: Ident,
  pub mode: BindingMode,
  pub is_mut: bool,
  pub span: Span,
}

impl BindingPat {
  pub fn new(
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
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    let mut mode = BindingMode::Move;
    let mut is_mut = false;
    let mut start_span = None;
    loop {
      let (kind, span) = cx.next_token()?;
      if start_span.is_none() {
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
pub enum Pattern {
  RefMut(RefMutPat),
  Ref(RefPat),
  Slice(SlicePat),
  Paren(ParenPat),
  Binding(BindingPat),
}

impl Pattern {
  pub fn span(&self) -> Span {
    match self {
      Pattern::RefMut(x) => x.span,
      Pattern::Ref(x) => x.span,
      Pattern::Slice(x) => x.span,
      Pattern::Paren(x) => x.span,
      Pattern::Binding(x) => x.span,
    }
  }

  pub fn span_min(&self) -> Span {
    match self {
      Pattern::RefMut(x) => x.span_min,
      Pattern::Ref(x) => x.span_min,
      Pattern::Slice(x) => x.span_min,
      Pattern::Paren(x) => x.span_min,
      Pattern::Binding(x) => x.span,
    }
  }

  #[must_use]
  pub fn map<F: Fn(Self) -> Self>(self, f: F) -> Self {
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
  pub fn simplify(self) -> Self {
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
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    use TokenKind::*;
    Ok(match cx.peek_expect()? {
      (Amp, _) => Pattern::Ref(RefPat::parse(cx)?),
      (AmpMut, _) => Pattern::RefMut(RefMutPat::parse(cx)?),
      (Ref, _) => Pattern::Binding(BindingPat::parse(cx)?),
      (RefMut, _) => Pattern::Binding(BindingPat::parse(cx)?),
      (Mut, _) => Pattern::Binding(BindingPat::parse(cx)?),
      (ParenStart, _) => Pattern::Paren(ParenPat::parse(cx)?),
      (SliceStart, _) => Pattern::Slice(SlicePat::parse(cx)?),
      (Binding(_), _) => Pattern::Binding(BindingPat::parse(cx)?),
      (_, span) => Err(Error::Unexpected(span))?,
    })
  }
}

// # Expressions

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeExpr {
  pub name: Ident,
  pub span: Span,
}

impl TypeExpr {
  pub fn new(name: Ident, span: Span) -> Self {
    Self { name, span }
  }
}

impl Display for TypeExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.name)
  }
}

impl Parse for TypeExpr {
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
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
pub struct RefMutExpr {
  pub expr: Box<Expr>,
  pub span: Span,
  pub span_min: Span,
}

impl RefMutExpr {
  pub fn new(expr: Expr, span: Span, span_min: Span) -> Self {
    Self { expr: Box::new(expr), span, span_min }
  }
}

impl Display for RefMutExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "&mut {}", self.expr)
  }
}

impl Parse for RefMutExpr {
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::AmpMut])?;
    let expr = Expr::parse(cx)?;
    let span = span1.extend_across(expr.span());
    Ok(Self::new(expr, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefExpr {
  pub expr: Box<Expr>,
  pub span: Span,
  pub span_min: Span,
}

impl RefExpr {
  pub fn new(expr: Expr, span: Span, span_min: Span) -> Self {
    Self { expr: Box::new(expr), span, span_min }
  }
}

impl Display for RefExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "&{}", self.expr)
  }
}

impl Parse for RefExpr {
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::Amp])?;
    let expr = Expr::parse(cx)?;
    let span = span1.extend_across(expr.span());
    Ok(Self::new(expr, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SliceExpr {
  pub expr: Box<Expr>,
  pub span: Span,
  pub span_min: Span,
}

impl SliceExpr {
  pub fn new(expr: Expr, span: Span, span_min: Span) -> Self {
    Self { expr: Box::new(expr), span, span_min }
  }
}

impl Display for SliceExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[{}]", self.expr)
  }
}

impl Parse for SliceExpr {
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::SliceStart])?;
    let expr = Expr::parse(cx)?;
    let span2 = cx.expect(&[TokenKind::SliceEnd])?;
    let span = span1.extend_across(span2);
    Ok(Self::new(expr, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParenExpr {
  pub expr: Box<Expr>,
  pub span: Span,
  pub span_min: Span,
}

impl ParenExpr {
  pub fn new(expr: Expr, span: Span, span_min: Span) -> Self {
    Self { expr: Box::new(expr), span, span_min }
  }
}

impl Display for ParenExpr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "({})", self.expr)
  }
}

impl Parse for ParenExpr {
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    let span1 = cx.expect(&[TokenKind::ParenStart])?;
    let expr = Expr::parse(cx)?;
    let span2 = cx.expect(&[TokenKind::ParenEnd])?;
    let span = span1.extend_across(span2);
    Ok(Self::new(expr, span, span1))
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Expr {
  Type(TypeExpr),
  RefMut(RefMutExpr),
  Ref(RefExpr),
  Slice(SliceExpr),
  Paren(ParenExpr),
}

impl Expr {
  pub fn span(&self) -> Span {
    match self {
      Expr::Type(x) => x.span,
      Expr::RefMut(x) => x.span,
      Expr::Ref(x) => x.span,
      Expr::Slice(x) => x.span,
      Expr::Paren(x) => x.span,
    }
  }

  pub fn span_min(&self) -> Span {
    match self {
      Expr::Type(x) => x.span,
      Expr::RefMut(x) => x.span_min,
      Expr::Ref(x) => x.span_min,
      Expr::Slice(x) => x.span_min,
      Expr::Paren(x) => x.span_min,
    }
  }

  #[must_use]
  pub fn map<F: Fn(Expr) -> Expr>(self, f: F) -> Self {
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
  pub fn simplify(self) -> Self {
    self.map(|expr| match expr {
      Expr::Type(_) => expr,
      Expr::RefMut(_) => expr,
      Expr::Ref(_) => expr,
      Expr::Slice(_) => expr,
      Expr::Paren(x) => *x.expr,
    })
  }

  #[must_use]
  pub fn make_shared(self) -> Self {
    match self {
      Expr::Type(_) => self,
      Expr::RefMut(RefMutExpr { expr, span, span_min }) => {
        Expr::Ref(RefExpr { expr, span, span_min })
      }
      Expr::Ref(_) => self,
      Expr::Slice(_) => self,
      Expr::Paren(_) => self,
    }
  }

  #[must_use]
  pub fn make_all_shared(self) -> Self {
    self.map(|expr| expr.make_shared())
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
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
    use TokenKind::*;
    Ok(match cx.peek_expect()? {
      (Amp, _) => Expr::Ref(RefExpr::parse(cx)?),
      (AmpMut, _) => Expr::RefMut(RefMutExpr::parse(cx)?),
      (ParenStart, _) => Expr::Paren(ParenExpr::parse(cx)?),
      (SliceStart, _) => Expr::Slice(SliceExpr::parse(cx)?),
      (Type(_), _) => Expr::Type(TypeExpr::parse(cx)?),
      (_, span) => Err(Error::Unexpected(span))?,
    })
  }
}

// # Statements

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LetStmt {
  pub pat: Pattern,
  pub expr: Expr,
  pub span: Span,
  pub span_min: Span,
}

impl LetStmt {
  pub fn new(
    pat: Pattern,
    expr: Expr,
    span: Span,
    span_min: Span,
  ) -> Self {
    Self { pat, expr, span, span_min }.simplify()
  }

  pub fn from_str(xs: &str) -> Result<LetStmt, Error> {
    let xs = lex(xs)?;
    let mut cx = Ctx::new(&xs);
    LetStmt::parse(&mut cx).map(|x| x.simplify())
  }

  pub fn from_pat_expr(pat: &Pattern, expr: &Expr) -> Self {
    let xs = format!("let {} = {};", pat, expr);
    Self::from_str(&xs).unwrap()
  }

  #[cfg(test)]
  pub fn as_string(&self) -> String {
    format!("{}", self)
  }

  #[must_use]
  pub fn simplify(self) -> Self {
    Self {
      pat: self.pat.simplify(),
      expr: self.expr.simplify(),
      span: self.span,
      span_min: self.span_min,
    }
  }

  #[allow(dead_code)]
  #[must_use]
  pub fn reparse(self) -> Self {
    let xs = lex(&format!("{}", self)).unwrap();
    let mut cx = Ctx::new(&xs);
    Self::parse(&mut cx).unwrap().simplify()
  }

  pub fn parsing_marks(&self) -> ParsingMarks<'_> {
    ParsingMarks(self)
  }
}

pub struct ParsingMarks<'s>(&'s LetStmt);

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
  fn parse(cx: &mut Ctx<'_>) -> PResult<Self> {
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

#[cfg(test)]
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

#[cfg(test)]
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

fn b64encode(xs: &[u8]) -> Vec<u8> {
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

#[cfg(test)]
fn b64decode(xs: &[u8]) -> Option<Vec<u8>> {
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
pub enum MermaidLinkKind {
  View,
  Edit,
  Img(&'static str),
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct Mermaid<C> {
  link_kind: MermaidLinkKind,
  code: C,
}

impl<C> Mermaid<C> {
  pub fn new(link_kind: MermaidLinkKind, code: C) -> Self {
    Self { link_kind, code }
  }
}

impl<C: Display> Display for Mermaid<C> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write_mermaid_link(f, &self.code, self.link_kind)
  }
}

pub fn write_mermaid_link<W: Write, C: Display>(
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
pub enum Node {
  Start,
  Error,
  Move,
  MoveBehindRef,
  MoveBehindRefMut,
  RefMut,
  RefMutBehindRef,
  Ref,
}

pub const NODES: &[Node] = {
  use Node::*;
  &[Move, MoveBehindRef, RefMut, RefMutBehindRef, Ref]
};

impl Node {
  pub fn label(&self) -> &'static str {
    match self {
      Node::Start => "Start",
      Node::Error => "Error",
      Node::Move => "move",
      Node::MoveBehindRef => "move behind &",
      Node::MoveBehindRefMut => "move behind &mut",
      Node::RefMut => "ref mut",
      Node::RefMutBehindRef => "ref mut behind &",
      Node::Ref => "ref",
    }
  }

  pub fn attr(&self) -> &'static str {
    match self {
      Node::Start => "start",
      Node::Error => "error",
      Node::Move => "move",
      Node::MoveBehindRef => "move_behind_shared_ref",
      Node::MoveBehindRefMut => "move_behind_mut_ref",
      Node::RefMut => "ref_mut",
      Node::RefMutBehindRef => "ref_mut_behind_shared_ref",
      Node::Ref => "ref",
    }
  }

  pub fn id(&self) -> &'static str {
    match self {
      Node::Start => "I",
      Node::Error => "E",
      Node::Move => "M",
      Node::MoveBehindRef => "N",
      Node::MoveBehindRefMut => "O",
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
pub enum EdgeDir {
  Forward,
  Backward,
  ToSelf,
  Error,
}

impl EdgeDir {
  pub fn color(&self) -> &'static str {
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
pub enum EdgeTy {
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
  BindingVsRefMut,
  Binding,
  Empty,
}

pub const EDGE_TYS: &[EdgeTy] = {
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
    BindingVsRefMut,
    Binding,
  ]
};

impl EdgeTy {
  pub fn label(&self) -> &'static str {
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
      EdgeTy::BindingVsRefMut => "x-&mut",
      EdgeTy::Binding => "x",
      EdgeTy::Empty => "",
    }
  }

  pub fn description(&self) -> &'static str {
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
      EdgeTy::BindingVsRefMut => {
        "binding pattern matches against mutable reference"
      },
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
pub struct Edge {
  pub from: Node,
  pub ty: EdgeTy,
  pub to: Node,
}

impl Edge {
  pub fn new(from: Node, to: Node, ty: EdgeTy) -> Self {
    Self { from, to, ty }
  }

  pub fn dir(&self) -> EdgeDir {
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
pub struct Graph {
  pub edges: Vec<Edge>,
}

impl Graph {
  pub fn new() -> Self {
    Self { edges: Vec::new() }
  }

  pub fn add_edge(&mut self, from: Node, to: Node, ty: EdgeTy) {
    self.edges.push(Edge::new(from, to, ty))
  }

  pub fn nodes(&self) -> Vec<Node> {
    let mut xs = HashSet::new();
    for x in &self.edges {
      xs.insert(x.from);
      xs.insert(x.to);
    }
    let mut xs = xs.drain().collect::<Vec<_>>();
    xs.sort();
    xs
  }

  pub fn sort_edges(&mut self) {
    self.edges.sort();
  }

  /// Delete edges from terminal nodes.
  pub fn simplify_terminal(&mut self) {
    let mut h = HashSet::new();
    for x in &self.edges {
      if x.from != x.to {
        h.insert(x.from);
      }
    }
    self.edges.retain(|x| h.contains(&x.from));
  }

  /// Delete edge types that always recurse.
  pub fn simplify_always_recurse(&mut self) {
    let mut h = HashSet::new();
    for x in &self.edges {
      if x.from != x.to {
        h.insert(x.ty);
      }
    }
    self.edges.retain(|x| h.contains(&x.ty));
  }

  /// Delete error edges.
  pub fn remove_error_edges(&mut self) {
    self.edges.retain(|x| x.to != Node::Error);
  }

  /// Delete recursive edges.
  #[allow(dead_code)]
  fn remove_recursive_edges(&mut self) {
    self.edges.retain(|x| x.from != x.to);
  }

  /// Delete redundant nodes.
  // A node is redundant if all of its non-self outgoing edges match
  // the outgoing edges of some other node.
  #[allow(dead_code)]
  fn remove_redundant_nodes(&mut self) {
    let mut m = HashMap::new();
    for x in &self.edges {
      if x.from != x.to {
        let s = m.entry(x.from).or_insert_with(HashSet::new);
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

  pub fn simplify_recuse(&mut self) {
    self.simplify_terminal();
    self.simplify_always_recurse();
    self.sort_edges();
  }

  pub fn simplify_error(&mut self) {
    self.remove_error_edges();
    self.sort_edges();
  }

  pub fn edge_idxs_by_dir(&self) -> HashMap<EdgeDir, HashSet<usize>> {
    let mut xs = HashMap::new();
    for (i, x) in self.edges.iter().enumerate() {
      let s = xs.entry(x.dir()).or_insert(HashSet::new());
      s.insert(i);
    }
    xs
  }

  pub fn write_link_styles<W: Write>(
    &self,
    w: &mut W,
  ) -> fmt::Result {
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

  pub fn write_header<W: Write>(&self, w: &mut W) -> fmt::Result {
    writeln!(
      w,
      "%%{{ init: {{ 'flowchart': {{ 'curve': 'linear' }} }} }}%%\n"
    )?;
    writeln!(w, "flowchart LR\n")
  }

  pub fn write_nodes<W: Write>(&self, w: &mut W) -> fmt::Result {
    for x in self.nodes() {
      writeln!(w, "{}", x)?;
    }
    writeln!(w)
  }

  pub fn write_edges<W: Write>(
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
pub struct Conf {
  /// Disable match ergonomics entirely.
  pub no_me: bool,
  /// When the default binding mode is not `move`, writing `mut` on a
  /// binding is an error.
  pub rule1: bool,
  /// When a reference pattern matches against a reference, do not
  /// update the default binding mode.
  pub rule2: bool,
  /// Keep track of whether we have matched either a reference pattern
  /// or a non-reference pattern against a shared reference, and if
  /// so, set the DBM to `ref` when we would otherwise set it to `ref
  /// mut`.
  pub rule3: bool,
  /// If we've previously matched against a shared reference in the
  /// scrutinee (or against a `ref` DBM under Rule 4, or against a
  /// mutable reference treated as a shared one or a `ref mut` DB
  /// treated as a `ref` one under Rule 5), if we've reached a binding
  /// and the scrutinee is a mutable reference, coerce it to a shared
  /// reference.
  pub rule3_ext1: bool,
  /// If a reference pattern is being matched against a non-reference
  /// type and if the DBM is `ref` or `ref mut`, match the pattern
  /// against the DBM as though it were a type.
  pub rule4: bool,
  /// If an `&` pattern is being matched against a non-reference type
  /// or an `&mut` pattern is being matched against a shared reference
  /// type or a non-reference type, and if the DBM is `ref` or `ref
  /// mut`, match the pattern against the DBM as though it were a
  /// type.
  pub rule4_ext: bool,
  /// If an `&` pattern is being matched against a mutable reference
  /// type or a non-reference type, or if an `&mut` pattern is being
  /// matched against a shared reference type or a non-reference type,
  /// and if the DBM is `ref` or `ref mut`, match the pattern against
  /// the DBM as though it were a type.
  pub rule4_ext2: bool,
  /// If the DBM is `ref` or `ref mut`, match a reference pattern
  /// against it as though it were a type *before* considering the
  /// scrutinee.
  pub rule4_early: bool,
  /// If a `&` pattern is being matched against a type of mutable
  /// reference (or against a `ref mut` DBM under *Rule 4*), act as
  /// though the type were a shared reference instead (or that a `ref
  /// mut` DBM is a `ref` DBM instead).
  pub rule5: bool,
  /// Rule 3, but lazily applied.
  pub rule3_lazy: bool,
  /// Spin rule.
  pub spin: bool,
}

impl Conf {
  #[allow(dead_code)]
  pub fn pre_rfc2005() -> Self {
    Self { no_me: true, ..<_>::default() }
  }

  pub fn rfc2005() -> Self {
    Self::default()
  }

  #[allow(dead_code)]
  pub fn rfc3627_2021() -> Self {
    Self {
      rule3: true,
      rule4: true,
      rule4_ext: true,
      rule5: true,
      ..<_>::default()
    }
  }

  #[allow(dead_code)]
  pub fn rfc_3627_2024_min() -> Self {
    Self { rule1: true, rule2: true, ..<_>::default() }
  }

  pub fn rfc3627_2024() -> Self {
    Self {
      rule1: true,
      rule2: true,
      rule3: true,
      rule4: true,
      rule4_ext: true,
      rule5: true,
      ..<_>::default()
    }
  }

  pub fn rpjohnst_2024() -> Self {
    Self {
      rule1: true,
      rule2: true,
      rule3_lazy: true,
      rule4_early: true,
      spin: true,
      ..<_>::default()
    }
  }

  pub fn waffle_2024() -> Self {
    Self {
      rule1: true,
      rule3: true,
      rule3_ext1: true,
      rule4_early: true,
      ..<_>::default()
    }
  }

  pub fn get_mut(
    &mut self,
    flag: &str,
  ) -> Result<&mut bool, ConfError> {
    Ok(match flag {
      "no_me" | "no_match_ergonomics" => &mut self.no_me,
      "rule1" => &mut self.rule1,
      "rule2" => &mut self.rule2,
      "rule3" => &mut self.rule3,
      "rule3_ext1" => &mut self.rule3_ext1,
      "rule3_lazy" => &mut self.rule3_lazy,
      "rule4" => &mut self.rule4,
      "rule4_early" => &mut self.rule4_early,
      "rule4_ext" => &mut self.rule4_ext,
      "rule4_ext2" => &mut self.rule4_ext2,
      "rule5" => &mut self.rule5,
      "spin" | "spin_rule" => &mut self.spin,
      _ => return Err(ConfError::UnknownFlag(flag.to_string())),
    })
  }

  pub fn set(&mut self, flag: &str) -> Result<(), ConfError> {
    match flag {
      "stable" => *self = Self::rfc2005(),
      "proposed" | "rfc" => *self = Self::rfc3627_2024(),
      "rpjohnst" => *self = Self::rpjohnst_2024(),
      "waffle" => *self = Self::waffle_2024(),
      _ => {
        let flag = self.get_mut(flag)?;
        *flag = true;
      }
    }
    Ok(())
  }

  pub fn unset(&mut self, flag: &str) -> Result<(), ConfError> {
    let flag = self.get_mut(flag)?;
    *flag = false;
    Ok(())
  }
}

impl Display for Conf {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "| no_match_ergonomics: {}", self.no_me)?;
    writeln!(f, "| rule1: {}", self.rule1)?;
    writeln!(f, "| rule2: {}", self.rule2)?;
    writeln!(f, "| rule3: {}", self.rule3)?;
    writeln!(f, "| rule3_ext1: {}", self.rule3_ext1)?;
    writeln!(f, "| rule3_lazy: {}", self.rule3_lazy)?;
    writeln!(f, "| rule4: {}", self.rule4)?;
    writeln!(f, "| rule4_early: {}", self.rule4_early)?;
    writeln!(f, "| rule4_ext2: {}", self.rule4_ext2)?;
    writeln!(f, "| rule4_ext: {}", self.rule4_ext)?;
    writeln!(f, "| rule5: {}", self.rule5)?;
    writeln!(f, "| spin_rule: {}", self.spin)?;
    Ok(())
  }
}

#[derive(
  Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd,
)]
#[repr(u8)]
pub enum StepNote {
  LazilySetToRef,
  EagerlySetToRef,
  SpinHappened,
  LeakToMove,
  MutResetsDBM,
  AppliedRule1,
  AppliedRule2,
  AppliedRule3,
  AppliedRule3Ext1,
  AppliedRule3Lazy,
  AppliedRule4,
  AppliedRule4Ext,
  AppliedRule4Ext2,
  AppliedRule4Early,
  AppliedRule5,
  AppliedSpin,
}

impl StepNote {
  pub fn description(&self) -> &'static str {
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
      StepNote::AppliedRule3Ext1 => {
        "apply Rule 3 ext1: treat `&mut` as `&` behind `&` at binding"
      }
      StepNote::AppliedRule3Lazy => {
        "apply lazy Rule 3: set DBM to `ref` behind `&` on bindings"
      }
      StepNote::AppliedRule4 => {
        "apply Rule 4: match `&/&mut` against DBM if no structural match"
      }
      StepNote::AppliedRule4Ext => {
        "apply Rule 4 ext: match `&/&mut` against DBM if no structural match"
      }
      StepNote::AppliedRule4Ext2 => {
        "apply Rule 4 ext2: match `&` against DBM overriding Rule 5"
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
pub enum StepError {
  MutTokNotAllowedBehindRef,
  RefMutNotAllowedBehindRef,
  CannotMoveBehindRef,
  CannotMoveBehindRefMut,
  MEDisabled,
  AlreadyError,
  TypeMismatch,
}

impl StepError {
  pub fn description(&self) -> &'static str {
    match self {
      StepError::MutTokNotAllowedBehindRef => {
        "`mut` on a binding is not allowed behind a reference"
      }
      StepError::RefMutNotAllowedBehindRef => {
        "cannot borrow as mutable"
      }
      StepError::CannotMoveBehindRef => {
        "cannot move out of a shared reference"
      }
      StepError::CannotMoveBehindRefMut => {
        "cannot move out of a mutable reference"
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
pub struct NodeStep {
  pub conf: Conf,
  pub node: Node,
  pub behind_ref: bool,
  pub behind_ref_mut: bool,
  pub notes: HashSet<StepNote>,
  pub error: Option<StepError>,
  pub last: bool,
  pub skip_expr: bool,
  pub skip_pat: bool,
  pub make_shared: bool,
  pub make_all_shared: bool,
  pub fill_shared: bool,
}

impl NodeStep {
  pub fn new(conf: Conf, node: Node) -> Self {
    Self {
      conf,
      node,
      behind_ref: false,
      behind_ref_mut: false,
      notes: HashSet::new(),
      error: None,
      last: false,
      skip_expr: false,
      skip_pat: false,
      make_shared: false,
      make_all_shared: false,
      fill_shared: false,
    }
  }

  pub fn add_note(&mut self, note: StepNote) {
    self.notes.insert(note);
  }

  pub fn node(&mut self, node: Node, note: StepNote) -> Self {
    assert!(!matches!(node, Node::Error));
    self.add_note(note);
    self.node = node;
    self.clone()
  }

  pub fn node_last(&mut self, node: Node, note: StepNote) -> Self {
    assert!(!matches!(node, Node::Error));
    self.add_note(note);
    self.node = node;
    self.last = true;
    self.clone()
  }

  pub fn err(&mut self, err: StepError) -> Self {
    self.error = Some(err);
    self.node = Node::Error;
    self.last = true;
    self.clone()
  }

  pub fn node1(&mut self, node: Node) -> Self {
    assert!(!matches!(node, Node::Error));
    self.node = node;
    self.clone()
  }

  pub fn node1_last(&mut self, node: Node) -> Self {
    assert!(!matches!(node, Node::Error));
    self.node = node;
    self.last = true;
    self.clone()
  }

  pub fn notes(&self) -> Vec<StepNote> {
    let mut xs = self.notes.iter().copied().collect::<Vec<_>>();
    xs.sort();
    xs
  }
}

pub fn node_step(
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
    rule3_ext1,
    rule4,
    rule4_ext,
    rule4_ext2,
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
    (MoveBehindRefMut, _) => {
      ns.behind_ref_mut = true;
      ns.node = Move;
    }
    (RefMutBehindRef, _) if rule3 => panic!(),
    (RefMutBehindRef, _) => {
      ns.behind_ref = true;
      ns.node = RefMut;
    }
    (Ref, _) => ns.behind_ref = true,
    _ => {}
  }
  match edge_ty {
    MutTok | RefMutTok | RefTok | BindingVsRefMut | Binding => {
      ns.skip_expr = true
    }
    NrpVsRef | NrpVsRefMut => ns.skip_pat = true,
    _ => {}
  }
  if no_me {
    match (ns.node, edge_ty) {
      (_, BindingVsRefMut | Binding) => {}
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
      (Ref, RefVsRefMut) if rule4_ext2 => {}
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
      (RefMut, RefMutVsRefMut | RefMutVsRef | RefMutVsT) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4Early);
        Some(ns.node1(Move))
      }
      (Ref, RefVsRef | RefVsRefMut | RefVsT) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4Early);
        Some(ns.node1(Move))
      }
      (RefMut, RefVsRef | RefVsRefMut | RefVsT) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4Early);
        Some(ns.err(TypeMismatch))
      }
      (Ref, RefMutVsRefMut | RefMutVsRef | RefMutVsT) => {
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
  if rule4_ext {
    next_node = match (ns.node, edge_ty) {
      (RefMut, RefMutVsRef) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4Ext);
        Some(ns.node1(Move))
      }
      _ => next_node,
    };
  }
  if rule4_ext2 {
    next_node = match (ns.node, edge_ty) {
      (Ref, RefVsRefMut) => {
        ns.skip_expr = true;
        ns.add_note(AppliedRule4Ext2);
        Some(ns.node1(Move))
      }
      _ => next_node,
    };
  }
  match edge_ty {
    NrpVsRef | RefVsRef | RefMutVsRef if !ns.skip_expr => {
      ns.behind_ref = true;
    }
    NrpVsRefMut | RefVsRefMut | RefMutVsRefMut if !ns.skip_expr => {
      ns.behind_ref_mut = true;
    }
    RefVsRef | RefVsRefMut | RefVsT if !ns.skip_pat => {
      ns.behind_ref = true;
    }
    RefMutVsRef | RefMutVsRefMut | RefMutVsT if !ns.skip_pat => {
      ns.behind_ref_mut = true;
    }
    _ => {}
  }
  let next_node = if let Some(next_node) = next_node {
    next_node
  } else {
    match (ns.node, edge_ty) {
      (
        Start | MoveBehindRef | MoveBehindRefMut | RefMutBehindRef,
        _,
      ) => unreachable!(),
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
      (node, BindingVsRefMut | Binding) => ns.node1_last(node),
    }
  };
  if ns.behind_ref {
    match (next_node.node, edge_ty) {
      (_, RefMutTok) => ns.err(RefMutNotAllowedBehindRef),
      (Move, BindingVsRefMut) if rule3_ext1 => {
        ns.make_shared = true;
        ns.node_last(MoveBehindRef, AppliedRule3Ext1)
      }
      (Move, BindingVsRefMut) => ns.err(CannotMoveBehindRef),
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
  } else if ns.behind_ref_mut {
    match (next_node.node, edge_ty) {
      (Move, BindingVsRefMut) => ns.err(CannotMoveBehindRefMut),
      (Move, _) => ns.node1(MoveBehindRefMut),
      _ => next_node,
    }
  } else {
    next_node
  }
}

pub fn walk_graph<F: FnMut(Node, EdgeTy, Node)>(
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

pub fn make_graph(conf: Conf) -> Graph {
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

pub fn edge_ty(pat: &Pattern, expr: &Expr) -> EdgeTy {
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
    (
      Pattern::Binding(BindingPat {
        mode: BindingMode::Move, ..
      }),
      Expr::RefMut(_),
    ) => BindingVsRefMut,
    (Pattern::Binding(BindingPat { mode, .. }), _) => match mode {
      BindingMode::Move => Binding,
      BindingMode::RefMut => RefMutTok,
      BindingMode::Ref => RefTok,
    },
  }
}

impl Pattern {
  pub fn reduce_one(&mut self) -> bool {
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
  pub fn reduce_one(&mut self) -> bool {
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
pub struct Reduction {
  pub conf: Conf,
  pub pat: Pattern,
  pub expr: Expr,
  pub node_step: NodeStep,
  pub edge_ty: EdgeTy,
  pub last: bool,
  pub dbm_applied: bool,
}

impl Reduction {
  pub fn new(conf: Conf, pat: Pattern, expr: Expr) -> Self {
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

  pub fn from_stmt(conf: Conf, stmt: LetStmt) -> Self {
    Self::new(conf, stmt.pat, stmt.expr)
  }

  #[allow(dead_code)]
  pub fn from_str(conf: Conf, xs: &str) -> Result<Self, Error> {
    Ok(Self::from_stmt(conf, LetStmt::from_str(xs)?))
  }

  pub fn step(&mut self) {
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
    if self.node_step.make_shared {
      self.expr = self.expr.clone().make_shared();
    }
    if self.node_step.make_all_shared {
      self.expr = self.expr.clone().make_all_shared();
    }
    self.edge_ty = edge_ty(&self.pat, &self.expr);
  }

  pub fn is_err(&self) -> bool {
    let x = matches!(self.node_step.node, Node::Error);
    let y = self.node_step.error.is_some();
    if x || y {
      assert!(x && y);
    }
    x
  }

  pub fn as_binding_mode(&self) -> BindingMode {
    match self.node_step.node {
      Node::Start | Node::Error => panic!(),
      Node::Move => BindingMode::Move,
      Node::MoveBehindRef => BindingMode::Move,
      Node::MoveBehindRefMut => BindingMode::Move,
      Node::RefMut => BindingMode::RefMut,
      Node::RefMutBehindRef => BindingMode::RefMut,
      Node::Ref => BindingMode::Ref,
    }
  }

  pub fn apply_dbm(&mut self) {
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

  pub fn as_stmt(&self) -> LetStmt {
    LetStmt::from_pat_expr(&self.pat, &self.expr)
  }

  pub fn as_type(&self) -> (Ident, Expr) {
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

  pub fn to_type(&self) -> Result<(Ident, Expr), StepError> {
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
      writeln!(f)?;
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
        if let Some(e) = r.node_step.error {
          writeln!(f, "//~{arrow} ERROR {e}")?;
        } else {
          writeln!(f, "//~{arrow} ERROR")?;
        }
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
    writeln!(f)?;
    Ok(())
  }
}

#[derive(Clone, Debug)]
pub struct ShowType<'a> {
  pub conf: Conf,
  pub stmt: &'a LetStmt,
}

impl<'a> ShowType<'a> {
  pub fn from_stmt(conf: Conf, stmt: &'a LetStmt) -> Self {
    Self { conf, stmt }
  }

  pub fn show_for<W: Write>(
    &self,
    f: &mut W,
    conf: Conf,
    label: &str,
  ) -> fmt::Result {
    let r = Reduction::from_stmt(conf, self.stmt.clone());
    write!(f, "//~| {}: ", label)?;
    match r.to_type() {
      Ok((ident, ty)) => writeln!(f, "{ident}: {ty}")?,
      Err(e) => writeln!(f, "ERROR {}", e)?,
    }
    Ok(())
  }

  pub fn show_for_current<W: Write>(
    &self,
    f: &mut W,
    label: &'static str,
  ) -> fmt::Result {
    let r = Reduction::from_stmt(self.conf, self.stmt.clone());
    write!(f, "//~| {}: ", label)?;
    match r.to_type() {
      Ok((ident, ty)) => writeln!(f, "{ident}: {ty}")?,
      Err(e) => writeln!(f, "ERROR {}", e)?,
    }
    Ok(())
  }
}

pub struct RuleGraph<'a> {
  pub conf: Conf,
  pub stmt: &'a LetStmt,
  pub top: RuleNode,
}

impl<'a> RuleGraph<'a> {
  pub fn new(conf: Conf, stmt: &'a LetStmt, top: RuleNode) -> Self {
    Self { conf, stmt, top }
  }
}

pub struct RuleNode {
  pub name: &'static str,
  pub rule: fn(&mut Conf),
  pub children: Vec<RuleNode>,
}

impl RuleNode {
  pub fn new(name: &'static str, rule: fn(&mut Conf)) -> Self {
    Self { name, rule, children: vec![] }
  }

  pub fn push(&mut self, child: RuleNode) {
    self.children.push(child);
  }

  #[must_use]
  pub fn add_rule(
    &mut self,
    name: &'static str,
    rule: fn(&mut Conf),
  ) -> &mut RuleNode {
    self.push(Self::new(name, rule));
    self.children.last_mut().unwrap()
  }
}

impl RuleNode {
  pub fn disp(
    &self,
    f: &mut fmt::Formatter<'_>,
    level: usize,
    mut conf: Conf,
    stmt: &LetStmt,
  ) -> fmt::Result {
    (self.rule)(&mut conf);
    let mut px = "".to_string();
    for _ in 0..level {
      px += "| ";
    }
    let name = format!("{px}* {}", self.name);
    let show = ShowType::from_stmt(conf, stmt);
    show.show_for(f, conf, &name)?;
    if self.children.is_empty() {
      return Ok(());
    }
    for (i, x) in self.children.iter().enumerate() {
      if i == self.children.len() - 1 {
        x.disp(f, level, conf, stmt)?;
      } else {
        writeln!(f, r"//~| {px}|\")?;
        x.disp(f, level + 1, conf, stmt)?;
      }
    }
    Ok(())
  }
}

impl<'a> Display for RuleGraph<'a> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.top.disp(f, 0, self.conf, self.stmt)
  }
}

impl<'a> Display for ShowType<'a> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if f.alternate() {
      writeln!(f, "{}", self.stmt)?;
    }
    self.show_for_current(f, "* As configured")?;
    writeln!(f, r"//~| -")?;
    let top = RuleNode::new("Rust stable == RFC 2005", |_| ());
    let mut graph = RuleGraph::new(Conf::default(), self.stmt, top);
    let node = &mut graph.top;
    {
      let node = node.add_rule("Rule 3", |x| x.rule3 = true);
      let node = node.add_rule("Rule 4", |x| x.rule4 = true);
      let _node = node
        .add_rule("Rule 5 == RFC 3627 Rust ..2021", |x| {
          x.rule5 = true
        });
    }
    let node = node.add_rule("Rule 1", |x| x.rule1 = true);
    {
      let node = node.add_rule("Rule 3", |x| x.rule3 = true);
      let node =
        node.add_rule("Rule 3 (ext1)", |x| x.rule3_ext1 = true);
      let _node = node.add_rule("Rule 4 (early) == waffle", |x| {
        x.rule4_early = true
      });
    }
    let node = node.add_rule("Rule 2", |x| x.rule2 = true);
    {
      let node =
        node.add_rule("Rule 3 (lazy)", |x| x.rule3_lazy = true);
      let node =
        node.add_rule("Rule 4 (early)", |x| x.rule4_early = true);
      let _node =
        node.add_rule("Spin rule == rpjohnst", |x| x.spin = true);
    }
    let node = node.add_rule("Rule 3", |x| x.rule3 = true);
    {
      let node =
        node.add_rule("Rule 3 (ext1)", |x| x.rule3_ext1 = true);
      {
        let node = node.add_rule("Rule 4", |x| x.rule4 = true);
        let node =
          node.add_rule("Rule 4 (ext)", |x| x.rule4_ext = true);
        let _node = node.add_rule("Rule 5", |x| x.rule5 = true);
      }
    }
    let node = node.add_rule("Rule 4", |x| x.rule4 = true);
    let node = node.add_rule("Rule 4 (ext)", |x| x.rule4_ext = true);
    {
      let node =
        node.add_rule("Rule 4 (ext2)", |x| x.rule4_ext2 = true);
      let _node = node.add_rule("Rule 5", |x| x.rule5 = true);
    }
    let _node = node
      .add_rule("Rule 5 == RFC 3627 Rust 2024", |x| x.rule5 = true);
    write!(f, "{graph}")?;
    Ok(())
  }
}
