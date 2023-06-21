function MarkdownHeadingFoldLevel()
  let level = getline(v:lnum)->matchstr('^=\+')->strlen()
  if level > 1
    return '>'..level
  endif
  return '='
endfunction

set foldmethod=expr
set foldexpr=MarkdownHeadingFoldLevel()
