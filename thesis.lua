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
  self:loadPackage("tableofcontents")
  self:loadPackage("bibtex")
  self:loadPackage("footnotes", {
      insertInto = "footnotes",
      stealFrom = { "content" }
    })
  SILE.call("font", { family = "Latin Modern Roman", size = "12pt" })
  SILE.call("neverindent")
  self:redefineCommands()
end

function class:redefineCommands ()
  local original_toc_fn = SILE.Commands["tableofcontents"]
  self:registerCommand("tableofcontents", function (options, _)
    SILE.call("thesis:chapterfont", _, function () original_toc_fn(options, _) end)
  end)

  self:registerCommand("tableofcontents:header", function (_, _)
    SILE.call("thesis:chapterfont", {}, function ()
      SILE.call("fluent", {}, { "tableofcontents-title" })
    end)
    SILE.call("medskip")
  end)

  self:registerCommand("tableofcontents:level1item", function (_, content)
    SILE.call("bigskip")
    local original_dotfill = SILE.Commands["dotfill"]
    SILE.Commands["dotfill"] = SILE.Commands["hfill"]  -- hack to have a blank hfill
    SILE.call("font", { size = 14, weight = 800 }, content)
    SILE.Commands["dotfill"] = original_dotfill
    SILE.call("medskip")
  end)
end

function class:registerCommands ()
  plain.registerCommands(self)

  local function sectioning(options, content)
    local number
    if SU.boolean(options.numbering, true) then
      SILE.call("increment-multilevel-counter", { id = "sectioning", level = options.level })
      number = self.packages.counters:formatMultilevelCounter(self:getMultilevelCounter("sectioning"))
      SILE.call("show-multilevel-counter", { id = "sectioning" })
    end
    if SU.boolean(options.toc, true) then
      SILE.call("tocentry", { level = options.level, number = number }, SU.subContent(content))
    end
    SILE.process( { " " } )
    SILE.process(content)
  end

  self:registerCommand("chapter", function (options, content)
    SILE.call("thesis:newpage")
    SILE.call("set-counter", { id = "footnote", value = 1 })
    SILE.call("thesis:chapterfont", {}, function ()
      sectioning({
        numbering = options.numbering,
        toc = options.toc,
        level = 1,
      }, content)
    end)
    SILE.call("par")
    SILE.call("bigskip")
  end, "Begin a new chapter")

  self:registerCommand("thesis:newpage", function (_, content)
    SILE.typesetter:leaveHmode()
    SILE.call("supereject")  -- prevent "underfull frame" warnings by vfilling page
    SILE.call("hbox")  -- prevents the skip to disappear at the top of the page
    SILE.call("skip", { height = "20mm" } )
  end)

  self:registerCommand("thesis:chapterfont", function (_, content)
    SILE.call("font", { family = "Latin Modern Sans", size = "24pt", weight = 10000 }, content)
  end)

  self:registerCommand("question", function (_, _) end)
  self:registerCommand("todo", function (_, _) end)
  self:registerCommand("citeauthor", function (_, _) end)

end

return class
