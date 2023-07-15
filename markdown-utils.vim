function MarkdownHeadingFoldLevel()
  let level = getline(v:lnum)->matchstr('^=\+')->strlen()
  if level > 1
    return '>'..level
  endif
  return '='
endfunction

set foldmethod=expr
set foldexpr=MarkdownHeadingFoldLevel()

"macro for text wrapping:
"wraps from cursor position to next . or :, advances cursor 1 word
let @w="gq/\\v(\\.|:)nW"

