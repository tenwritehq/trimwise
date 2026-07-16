(() => {
  "use strict";

  const COPY_RESET_DELAY_MS = 1800;
  const VIEWER_URL_REVOKE_DELAY_MS = 60_000;

  const buildPrompt = (markdownUrl) =>
    `Read this Trimwise documentation page and use it as the source when answering my questions: ${markdownUrl}`;

  const buildTargetUrl = (target, prompt) => {
    const encodedPrompt = encodeURIComponent(prompt);

    switch (target) {
      case "chatgpt":
        return `https://chatgpt.com/?q=${encodedPrompt}`;
      case "claude":
        return `https://claude.ai/new?q=${encodedPrompt}`;
      default:
        throw new Error(`Unsupported LLM target: ${target}`);
    }
  };

  const getMarkdownUrl = (root) => {
    const rawUrl = root.dataset.markdownUrl;
    if (!rawUrl) {
      throw new Error("Missing Markdown URL.");
    }
    return new URL(rawUrl, window.location.href);
  };

  const fetchMarkdown = async (root) => {
    const markdownUrl = getMarkdownUrl(root);
    const response = await fetch(markdownUrl, {
      credentials: "same-origin",
      headers: {
        Accept: "text/markdown, text/plain;q=0.9, */*;q=0.1",
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch Markdown: HTTP ${response.status}`);
    }

    const markdown = await response.text();
    if (!markdown.trim()) {
      throw new Error("The Markdown response was empty.");
    }

    return markdown;
  };

  const writeClipboard = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.inset = "0";
    textarea.style.opacity = "0";
    textarea.style.pointerEvents = "none";
    document.body.appendChild(textarea);
    textarea.select();

    const copied = document.execCommand("copy");
    textarea.remove();

    if (!copied) {
      throw new Error("The browser refused the clipboard operation.");
    }
  };

  const setCopyStatus = (root, label) => {
    for (const element of root.querySelectorAll("[data-copy-label]")) {
      element.textContent = label;
    }
  };

  const closeMenu = (root) => {
    const details = root.querySelector("details");
    if (details instanceof HTMLDetailsElement) {
      details.open = false;
    }
  };

  const copyPage = async (root) => {
    const markdown = await fetchMarkdown(root);
    await writeClipboard(markdown);
  };

  const viewMarkdown = async (root) => {
    /* Open synchronously so popup blockers treat it as user initiated. */
    const viewer = window.open("", "_blank");
    if (!viewer) {
      throw new Error("The browser blocked the Markdown viewer tab.");
    }

    viewer.opener = null;
    viewer.document.title = "Loading Markdown...";
    viewer.document.body.textContent = "Loading Markdown...";

    try {
      const markdown = await fetchMarkdown(root);
      const blob = new Blob([markdown], {
        type: "text/plain;charset=utf-8",
      });
      const blobUrl = URL.createObjectURL(blob);
      viewer.location.replace(blobUrl);
      window.setTimeout(
        () => URL.revokeObjectURL(blobUrl),
        VIEWER_URL_REVOKE_DELAY_MS,
      );
    } catch (error) {
      viewer.document.title = "Unable to open Markdown";
      viewer.document.body.textContent =
        error instanceof Error ? error.message : "Unable to open Markdown.";
      throw error;
    }
  };

  const initialiseRoot = (root) => {
    let markdownUrl;
    try {
      markdownUrl = getMarkdownUrl(root).href;
    } catch (error) {
      console.error(error);
      return;
    }

    const prompt = buildPrompt(markdownUrl);

    for (const link of root.querySelectorAll("[data-llm-target]")) {
      const target = link.dataset.llmTarget;
      try {
        link.href = buildTargetUrl(target, prompt);
      } catch (error) {
        console.error(error);
        link.removeAttribute("href");
        link.setAttribute("aria-disabled", "true");
      }
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    for (const root of document.querySelectorAll("[data-llm-page-actions]")) {
      initialiseRoot(root);
    }
  });

  document.addEventListener("click", async (event) => {
    const copyTrigger = event.target.closest("[data-copy-page]");
    if (!copyTrigger) {
      return;
    }

    const root = copyTrigger.closest("[data-llm-page-actions]");
    if (!root || copyTrigger.disabled) {
      return;
    }

    copyTrigger.disabled = true;
    setCopyStatus(root, "Copying...");

    try {
      await copyPage(root);
      setCopyStatus(root, "Copied!");
      closeMenu(root);
    } catch (error) {
      console.error("Unable to copy documentation page:", error);
      setCopyStatus(root, "Copy failed");
    } finally {
      window.setTimeout(() => {
        setCopyStatus(root, "Copy page");
        copyTrigger.disabled = false;
      }, COPY_RESET_DELAY_MS);
    }
  });

  document.addEventListener("click", async (event) => {
    const viewTrigger = event.target.closest("[data-view-markdown]");
    if (!viewTrigger) {
      return;
    }

    const root = viewTrigger.closest("[data-llm-page-actions]");
    if (!root || viewTrigger.disabled) {
      return;
    }

    viewTrigger.disabled = true;
    closeMenu(root);

    try {
      await viewMarkdown(root);
    } catch (error) {
      console.error("Unable to view Markdown:", error);
    } finally {
      viewTrigger.disabled = false;
    }
  });

  document.addEventListener("click", (event) => {
    const action = event.target.closest(
      "[data-download-markdown], [data-llm-target]",
    );
    if (action) {
      const root = action.closest("[data-llm-page-actions]");
      if (root) {
        closeMenu(root);
      }
    }
  });

  document.addEventListener("click", (event) => {
    for (const details of document.querySelectorAll(".llm-page-menu[open]")) {
      if (!details.contains(event.target)) {
        details.open = false;
      }
    }
  });
})();
