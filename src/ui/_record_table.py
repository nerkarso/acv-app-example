from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

_HTML = "<div id='root'></div>"

_JS = """
export default function (component) {
  const { data, parentElement, setStateValue, setTriggerValue } = component
  const root = parentElement.querySelector("#root")
  if (!root) return

  const columns = data.columns || []
  const rows = data.rows || []
  const selected = new Set(data.selected || [])
  const sort = data.sort || {}

  const style = document.createElement("style")
  style.textContent = `
    table { width: 100%; border-collapse: collapse; font-family: var(--st-font); color: var(--st-text-color); font-size: var(--st-base-font-size); }
    th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--st-border-color); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    th { background: var(--st-dataframe-header-background-color); font-weight: 600; }
    th.sortable { cursor: pointer; user-select: none; }
    th.sortable:hover { color: var(--st-primary-color); }
    tbody tr:hover td { background: var(--st-secondary-background-color); }
    button.view-btn { background: transparent; border: 1px solid var(--st-widget-border-color); color: var(--st-primary-color); border-radius: var(--st-button-radius); padding: 4px 12px; cursor: pointer; font-family: var(--st-font); font-size: var(--st-base-font-size); transition: opacity 0.2s; }
    button.view-btn:hover { opacity: 0.8; }
    input[type=checkbox] { cursor: pointer; }
  `

  const table = document.createElement("table")

  const thead = document.createElement("thead")
  const headRow = document.createElement("tr")
  const thCheck = document.createElement("th")
  thCheck.style.width = "32px"
  if (rows.length > 0) {
    const allChecked = rows.every((row) => selected.has(row.id))
    const someChecked = rows.some((row) => selected.has(row.id))
    const headCb = document.createElement("input")
    headCb.type = "checkbox"
    headCb.checked = allChecked
    headCb.indeterminate = someChecked && !allChecked
    headCb.onchange = () => {
      rows.forEach((row) => {
        if (headCb.checked) selected.add(row.id)
        else selected.delete(row.id)
      })
      setStateValue("selected", Array.from(selected))
    }
    thCheck.appendChild(headCb)
  }
  headRow.appendChild(thCheck)
  columns.forEach((col) => {
    const th = document.createElement("th")
    const arrow = sort.key === col.key ? (sort.dir === "desc" ? " ▼" : " ▲") : ""
    th.textContent = col.label + arrow
    if (col.width) th.style.width = col.width + "px"
    if (col.sortable !== false) {
      th.classList.add("sortable")
      th.onclick = () => setTriggerValue("sort_requested", col.key)
    }
    headRow.appendChild(th)
  })
  const thAction = document.createElement("th")
  thAction.style.width = "90px"
  headRow.appendChild(thAction)
  thead.appendChild(headRow)
  table.appendChild(thead)

  const tbody = document.createElement("tbody")
  rows.forEach((row) => {
    const tr = document.createElement("tr")

    const tdCheck = document.createElement("td")
    const cb = document.createElement("input")
    cb.type = "checkbox"
    cb.checked = selected.has(row.id)
    cb.onchange = () => {
      if (cb.checked) selected.add(row.id)
      else selected.delete(row.id)
      setStateValue("selected", Array.from(selected))
    }
    tdCheck.appendChild(cb)
    tr.appendChild(tdCheck)

    columns.forEach((col) => {
      const td = document.createElement("td")
      td.textContent = row[col.key] ?? ""
      tr.appendChild(td)
    })

    const tdAction = document.createElement("td")
    const btn = document.createElement("button")
    btn.type = "button"
    btn.className = "view-btn"
    btn.textContent = "View"
    btn.onclick = () => setTriggerValue("opened", row.id)
    tdAction.appendChild(btn)
    tr.appendChild(tdAction)

    tbody.appendChild(tr)
  })
  table.appendChild(tbody)

  root.replaceChildren(style, table)
}
"""

_RECORD_TABLE = st.components.v2.component(  # pyright: ignore[reportAttributeAccessIssue]
    "record_table", html=_HTML, js=_JS
)


def record_table(
    rows: list[dict[str, Any]],
    columns: list[dict[str, Any]],
    *,
    key: str,
    sort: dict[str, str] | None = None,
    on_opened_change: Callable[[], None] | None = None,
    on_selected_change: Callable[[], None] | None = None,
    on_sort_requested_change: Callable[[], None] | None = None,
) -> Any:
    component_state = st.session_state.get(key, {})
    selected = component_state.get("selected", [])

    return _RECORD_TABLE(
        key=key,
        data={"rows": rows, "columns": columns, "selected": selected, "sort": sort or {}},
        on_opened_change=on_opened_change or (lambda: None),
        on_selected_change=on_selected_change or (lambda: None),
        on_sort_requested_change=on_sort_requested_change or (lambda: None),
    )
