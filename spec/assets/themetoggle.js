(() => {
  const state = window.localStorage.invertTheme || ""
  const label = document.createElement("label")
  label.id = "theme-toggle";
  const checkbox = document.createElement("input")
  checkbox.type = "checkbox"
  checkbox.switch = true;
  checkbox.checked = state;
  checkbox.ariaLabel = "Invert the color scheme from the system default"
  label.appendChild(checkbox)
  label.appendChild(document.createElement("span"))
  document.querySelector(".headerButtons").appendChild(label)

  function handler() {
    if (document.querySelector("#theme-toggle input").checked) {
      window.localStorage.invertTheme = "1"
      document.documentElement.dataset.invertTheme = ""
    } else {
      window.localStorage.removeItem("invertTheme")
      delete document.documentElement.dataset.invertTheme
    }
  }
  handler()

  document.querySelector("#theme-toggle input").addEventListener("change", handler)
})()

