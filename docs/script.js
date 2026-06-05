/* =====================================================
   AtrionNet Documentation — script.js
   ===================================================== */

document.addEventListener('DOMContentLoaded', () => {

  // ─── Init highlight.js ──────────────────────────────
  if (window.hljs) {
    document.querySelectorAll('pre code').forEach(block => {
      hljs.highlightElement(block);
    });
  }

  // ─── State ──────────────────────────────────────────
  const sidebar    = document.getElementById('sidebar');
  const overlay    = document.getElementById('overlay');
  const hamburger  = document.getElementById('hamburger');
  const themeToggle = document.getElementById('themeToggle');
  const searchInput = document.getElementById('searchInput');
  const searchResults = document.getElementById('searchResults');

  // ─── Theme Toggle ───────────────────────────────────
  const savedTheme = localStorage.getItem('atrionnet-theme') || 'dark';
  applyTheme(savedTheme);

  themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    localStorage.setItem('atrionnet-theme', next);
  });

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    themeToggle.textContent = theme === 'dark' ? '🌙' : '☀️';
  }

  // ─── Mobile Sidebar ─────────────────────────────────
  hamburger.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    overlay.classList.toggle('active');
  });

  overlay.addEventListener('click', () => {
    sidebar.classList.remove('open');
    overlay.classList.remove('active');
  });

  // ─── Sidebar Navigation ─────────────────────────────
  const sidebarLinks = document.querySelectorAll('.sidebar-link[data-target]');

  sidebarLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.getAttribute('data-target');
      const section = document.getElementById(target);
      if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        // Close mobile sidebar
        sidebar.classList.remove('open');
        overlay.classList.remove('active');
      }
    });
  });

  // ─── Active Section Tracking (IntersectionObserver) ─
  const sections = document.querySelectorAll('.doc-section');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        setActiveLink(id);
      }
    });
  }, {
    rootMargin: `-${58 + 20}px 0px -60% 0px`,
    threshold: 0
  });

  sections.forEach(section => observer.observe(section));

  function setActiveLink(id) {
    sidebarLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('data-target') === id) {
        link.classList.add('active');
        // Scroll active link into view in sidebar
        link.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    });
  }

  // ─── Copy Code Buttons ───────────────────────────────
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const codeBlock = btn.closest('.code-block');
      const code = codeBlock.querySelector('pre code') || codeBlock.querySelector('pre');
      const text = code ? code.innerText : '';

      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = '✓ Copied';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = 'Copy';
          btn.classList.remove('copied');
        }, 2000);
      }).catch(() => {
        // Fallback for browsers that don't support clipboard API
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        btn.textContent = '✓ Copied';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = 'Copy';
          btn.classList.remove('copied');
        }, 2000);
      });
    });
  });

  // ─── Search ─────────────────────────────────────────

  // Build a search index from all sections
  const searchIndex = buildSearchIndex();

  searchInput.addEventListener('input', debounce(handleSearch, 200));
  searchInput.addEventListener('focus', () => {
    if (searchInput.value.trim()) handleSearch();
  });

  // Close search results on outside click
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-box') && !e.target.closest('.search-results')) {
      searchResults.classList.remove('visible');
    }
  });

  // Close on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      searchResults.classList.remove('visible');
      searchInput.blur();
    }
  });

  // Keyboard shortcut: / to focus search
  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement !== searchInput) {
      e.preventDefault();
      searchInput.focus();
    }
  });

  function handleSearch() {
    const query = searchInput.value.trim().toLowerCase();
    if (!query) {
      searchResults.classList.remove('visible');
      return;
    }

    const results = searchIndex.filter(item =>
      item.title.toLowerCase().includes(query) ||
      item.text.toLowerCase().includes(query)
    ).slice(0, 8);

    if (results.length === 0) {
      searchResults.innerHTML = `
        <div class="search-result-item">
          <h4 style="color:var(--text-3)">No results for "${escapeHtml(query)}"</h4>
          <p>Try a different search term.</p>
        </div>`;
    } else {
      searchResults.innerHTML = results.map(r => `
        <div class="search-result-item" data-target="${r.id}">
          <h4>${highlight(r.title, query)}</h4>
          <p>${highlight(truncate(r.text, 90), query)}</p>
        </div>
      `).join('');

      searchResults.querySelectorAll('.search-result-item').forEach(item => {
        item.addEventListener('click', () => {
          const targetId = item.getAttribute('data-target');
          const section = document.getElementById(targetId);
          if (section) {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
            setActiveLink(targetId);
          }
          searchResults.classList.remove('visible');
          searchInput.value = '';
        });
      });
    }

    searchResults.classList.add('visible');
  }

  function buildSearchIndex() {
    const index = [];
    document.querySelectorAll('.doc-section').forEach(section => {
      const id = section.id;
      const h1 = section.querySelector('h1');
      const title = h1 ? h1.textContent : id;

      // Gather all paragraph text in this section
      const paragraphs = section.querySelectorAll('p, li, h2, h3');
      let text = Array.from(paragraphs)
        .map(el => el.textContent)
        .join(' ')
        .replace(/\s+/g, ' ')
        .trim();

      index.push({ id, title, text });

      // Also index sub-headings as separate entries
      section.querySelectorAll('h2, h3').forEach(heading => {
        index.push({
          id,
          title: `${title} › ${heading.textContent}`,
          text: heading.textContent
        });
      });
    });
    return index;
  }

  function highlight(text, query) {
    if (!query) return escapeHtml(text);
    const escaped = escapeRegex(query);
    return escapeHtml(text).replace(
      new RegExp(`(${escaped})`, 'gi'),
      '<mark style="background:rgba(88,166,255,0.25);color:var(--blue);border-radius:2px;padding:0 2px">$1</mark>'
    );
  }

  function truncate(str, len) {
    return str.length > len ? str.slice(0, len) + '…' : str;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function debounce(fn, delay) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
  }

  // ─── Smooth link anchors ─────────────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const targetId = anchor.getAttribute('href').slice(1);
      const target = document.getElementById(targetId);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ─── Endpoint Collapsible (optional) ────────────────
  document.querySelectorAll('.endpoint-header').forEach(header => {
    header.style.cursor = 'pointer';
    const body = header.nextElementSibling;
    if (!body || !body.classList.contains('endpoint-body')) return;

    // Start open
    body.style.display = 'block';

    // Add toggle indicator
    const indicator = document.createElement('span');
    indicator.textContent = '▾';
    indicator.style.cssText = 'margin-left:auto;color:var(--text-3);font-size:14px;transition:transform 0.2s';
    header.appendChild(indicator);

    header.addEventListener('click', () => {
      const isOpen = body.style.display !== 'none';
      body.style.display = isOpen ? 'none' : 'block';
      indicator.style.transform = isOpen ? 'rotate(-90deg)' : 'rotate(0deg)';
    });
  });

  // ─── Animate sections on entry ──────────────────────
  const fadeObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        fadeObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.05 });

  sections.forEach(section => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(16px)';
    section.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    fadeObserver.observe(section);
  });

  // Immediately show the first visible section
  const firstSection = document.querySelector('.doc-section');
  if (firstSection) {
    firstSection.style.opacity = '1';
    firstSection.style.transform = 'translateY(0)';
  }

  // ─── Table of Contents Active Highlight ─────────────
  // Add subtle line numbers to code blocks
  document.querySelectorAll('pre code').forEach(block => {
    // Skip diagram/text blocks
    const lang = block.className;
    if (lang.includes('diagram') || lang === '') return;
  });

  // ─── Metric Card Counters ────────────────────────────
  const metricValues = document.querySelectorAll('.metric-value');

  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const finalText = el.textContent.trim();
        const num = parseFloat(finalText.replace(/[^0-9.]/g, ''));

        if (!isNaN(num) && num > 1) {
          animateCounter(el, 0, num, finalText);
        }
        counterObserver.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  metricValues.forEach(mv => counterObserver.observe(mv));

  function animateCounter(el, start, end, originalText) {
    const duration = 800;
    const startTime = performance.now();
    const suffix = originalText.replace(/[0-9.]/g, '').trim();

    const frame = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const current = Math.round(start + (end - start) * eased);
      el.textContent = current + suffix;
      if (progress < 1) requestAnimationFrame(frame);
      else el.textContent = originalText;
    };

    requestAnimationFrame(frame);
  }

  // ─── Back to Top ─────────────────────────────────────
  const backToTop = document.createElement('button');
  backToTop.innerHTML = '↑';
  backToTop.title = 'Back to top';
  backToTop.style.cssText = `
    position: fixed;
    bottom: 28px;
    right: 28px;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--blue-dark);
    color: white;
    border: none;
    font-size: 18px;
    cursor: pointer;
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 90;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    transition: opacity 0.2s, transform 0.2s;
  `;
  document.body.appendChild(backToTop);

  window.addEventListener('scroll', () => {
    if (window.scrollY > 400) {
      backToTop.style.display = 'flex';
    } else {
      backToTop.style.display = 'none';
    }
  });

  backToTop.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  backToTop.addEventListener('mouseenter', () => {
    backToTop.style.transform = 'scale(1.1)';
  });
  backToTop.addEventListener('mouseleave', () => {
    backToTop.style.transform = 'scale(1)';
  });

  // ─── Progress Bar ─────────────────────────────────────
  const progressBar = document.createElement('div');
  progressBar.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--blue), var(--purple));
    z-index: 200;
    transition: width 0.1s ease;
    pointer-events: none;
  `;
  document.body.appendChild(progressBar);

  window.addEventListener('scroll', () => {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    progressBar.style.width = `${progress}%`;
  });

  // ─── Keyboard shortcuts panel (press ?) ─────────────
  document.addEventListener('keydown', (e) => {
    if (e.key === '?' && document.activeElement !== searchInput) {
      showShortcutsModal();
    }
  });

  function showShortcutsModal() {
    const existing = document.getElementById('shortcuts-modal');
    if (existing) { existing.remove(); return; }

    const modal = document.createElement('div');
    modal.id = 'shortcuts-modal';
    modal.style.cssText = `
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.7);
      z-index: 300;
      display: flex;
      align-items: center;
      justify-content: center;
    `;
    modal.innerHTML = `
      <div style="background:var(--bg-3);border:1px solid var(--border-light);border-radius:12px;padding:32px;min-width:300px;max-width:400px;box-shadow:var(--shadow)">
        <h3 style="color:var(--text);margin-bottom:20px;font-size:16px">Keyboard Shortcuts</h3>
        <div style="display:flex;flex-direction:column;gap:10px">
          ${[
            ['/', 'Focus search'],
            ['Esc', 'Close search / modal'],
            ['?', 'Toggle this panel'],
          ].map(([key, desc]) => `
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span style="color:var(--text-3);font-size:13px">${desc}</span>
              <kbd style="background:var(--bg-4);border:1px solid var(--border-light);border-radius:4px;padding:2px 8px;font-family:JetBrains Mono,monospace;font-size:12px;color:var(--text-2)">${key}</kbd>
            </div>
          `).join('')}
        </div>
        <p style="margin-top:20px;font-size:12px;color:var(--text-muted);text-align:center">Click anywhere or press Esc to close</p>
      </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });
    document.addEventListener('keydown', function handler(e) {
      if (e.key === 'Escape' || e.key === '?') {
        modal.remove();
        document.removeEventListener('keydown', handler);
      }
    });
  }

  // ─── Print-friendly mode ─────────────────────────────
  window.addEventListener('beforeprint', () => {
    sections.forEach(s => {
      s.style.opacity = '1';
      s.style.transform = 'none';
    });
  });

  // ─── Table row hover enhancement ────────────────────
  document.querySelectorAll('tbody tr').forEach(row => {
    row.addEventListener('mouseenter', () => {
      row.style.transition = 'background 0.15s ease';
    });
  });

  // ─── Code block filename tooltip ────────────────────
  document.querySelectorAll('.code-filename').forEach(el => {
    el.title = `File: ${el.textContent}`;
  });

  // ─── External links — open in new tab ───────────────
  document.querySelectorAll('a[href^="http"]').forEach(link => {
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');
  });

  console.log('%cAtrionNet Docs', 'color:#58a6ff;font-size:20px;font-weight:700');
  console.log('%cPress / to search  |  ? for keyboard shortcuts', 'color:#8b949e;font-size:12px');
});
