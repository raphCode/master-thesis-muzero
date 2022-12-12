local plain = require("classes.plain")

local class = pl.class(plain)
class._name = "thesis"

class.defaultFrameset = {
  content = {
    left = "12.5%pw",
    right = "100%pw - 12.5%pw",
    top = "8.7%ph",
    bottom = "top(footnotes)"
  },
  folio = {
    left = "left(content)",
    right = "right(content)",
    top = "bottom(footnotes) + 5%ph",
    bottom = "top(folio) + 1cm"
  },
  footnotes = {
    left = "left(content)",
    right = "right(content)",
    height = "0",
    bottom = "100%ph - 16.5%ph"
  }
}

function class:_init(options)
  plain._init(self, options)
  self:loadPackage("footnotes", {
      insertInto = "footnotes",
      stealFrom = { "content" }
    })
  SILE.call("font", { family = "Latin Modern Roman", size = "12pt" })
  SILE.call("neverindent")
end

return class
