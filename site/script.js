const simulationTasks = [
  "Pick up the black bowl next to the plate and place it on the plate",
  "Pick up the black bowl next to the cookie box and place it on the plate",
  "Pick up the black bowl on the cookie box and place it on the plate",
  "Pick up the ketchup and place it in the basket",
  "Pick up the alphabet soup and place it in the basket",
  "Push the plate to the front of the stove",
  "Put the bowl on top of the cabinet",
  "Put the cream cheese in the bowl",
  "Put the wine bottle on top of the cabinet",
  "Put the wine bottle on the rack"
];

const realWorldTasks = [
  "Pick up the block and place it in the box",
  "Stack the red block on top of the blue block",
  "Stack the two blocks on the left on top of the block on the right",
  "Insert the three-prong plug into the socket",
  "Push the box to the designated location",
  "Pick up the water bottle and place it in the box"
];

const simulationResults = {
  minivla: {
    lt: [68.4, 49.6, 62.2, 55.2, 46.7, 40.5, 36.7, 10.5, 38.1, 0.0],
    apa: [72.6, 67.3, 66.1, 70.0, 56.5, 44.5, 37.2, 11.8, 20.9, 1.3],
    ecl: [80.0, 68.9, 74.7, 83.5, 64.7, 48.0, 39.7, 13.7, 54.3, 2.4]
  }
};

const performanceGroups = document.querySelector("[data-performance-groups]");
const renderPerformanceChart = (model) => {
  const results = simulationResults[model];
  performanceGroups.innerHTML = results.lt.map((lt, task) => {
    const bars = [
      ["lt", "LT", lt],
      ["apa", "APA", results.apa[task]],
      ["ecl", "ECL", results.ecl[task]]
    ].map(([className, method, value]) => `<i class="performance-bar ${className}" style="--value:${value}" data-value="${value.toFixed(1)}" title="Task ${task} · ${method}: ${value.toFixed(1)}%" aria-label="Task ${task}, ${method}, ${value.toFixed(1)} percent"></i>`).join("");
    return `<div class="performance-group"><div class="performance-bars">${bars}</div><span class="performance-task-label">T${task}</span></div>`;
  }).join("");
};

renderPerformanceChart("minivla");

const galleryConfig = {
  "simulation-minivla": { tasks: simulationTasks, prefix: "minivla-task", baseline: "-apa.gif", ecl: "-ecl.gif", baselineLabel: "APA", model: "MiniVLA", type: "image" },
  "simulation-pi05": { tasks: simulationTasks, prefix: "pi05-task", baseline: "-apa.gif", ecl: "-ecl.gif", baselineLabel: "APA", model: "π₀.₅", type: "image" },
  "piper-minivla": { tasks: realWorldTasks, prefix: "minivla-piper-task", baseline: "-apa.mp4", ecl: "-ecl.mp4", baselineLabel: "APA", model: "MiniVLA", type: "video" },
  "piper-pi05": { tasks: realWorldTasks, prefix: "pi05-piper-task", baseline: "-apa.mp4", ecl: "-ecl.mp4", baselineLabel: "APA", model: "π₀.₅", type: "video" },
  "xarm-minivla": { tasks: realWorldTasks, prefix: "minivla-xarm-task", baseline: "-apa.mp4", ecl: "-ecl.mp4", baselineLabel: "APA", model: "MiniVLA", type: "video" },
  "xarm-pi05": { tasks: realWorldTasks, prefix: "pi05-xarm-task", baseline: "-apa.mp4", ecl: "-ecl.mp4", baselineLabel: "APA", model: "π₀.₅", type: "video" }
};

document.querySelectorAll("[data-gallery]").forEach((gallery) => {
  const config = galleryConfig[gallery.dataset.gallery];
  gallery.innerHTML = config.tasks.map((task, index) => {
    const id = String(index).padStart(2, "0");
    const baselineSource = `assets/demos/${config.prefix}${id}${config.baseline}`;
    const eclSource = `assets/demos/${config.prefix}${id}${config.ecl}`;
    const media = (source, label) => config.type === "video"
      ? `<video muted loop playsinline preload="metadata" src="${source}" aria-label="${label}: ${task}"></video>`
      : `<img loading="lazy" src="${source}" alt="${label}: ${task}">`;
    return `<article class="task-demo-card"><div class="task-card-head"><span>Task ${index}</span><h3>${task}</h3></div><div class="task-pair-media"><figure class="demo-tile"><span>${config.baselineLabel}</span>${media(baselineSource, config.baselineLabel)}</figure><figure class="demo-tile ours"><span>ECL (Ours)</span>${media(eclSource, "ECL")}</figure></div><div class="task-meta"><p>${config.model} · Baseline / ECL</p></div></article>`;
  }).join("");
});

const galleryVideoObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.play().catch(() => {});
    } else {
      entry.target.pause();
    }
  });
}, { rootMargin: "180px 0px" });

document.querySelectorAll(".task-gallery video").forEach((video) => galleryVideoObserver.observe(video));

document.querySelectorAll(".demo-module").forEach((module) => {
  const tabs = module.querySelectorAll(".tab[data-tab]");
  const panels = module.querySelectorAll(".model-panel[data-panel]");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const selectedModel = tab.dataset.tab;
      tabs.forEach((item) => {
        const selected = item === tab;
        item.classList.toggle("active", selected);
        item.setAttribute("aria-selected", String(selected));
      });
      panels.forEach((panel) => {
        panel.hidden = panel.dataset.panel !== selectedModel;
      });
    });
  });
});
