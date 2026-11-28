const TROUPEKIT_THEMES = ['rust-dark', 'rust'];
const TROUPEKIT_STOCK_THEMES = ['light', 'coal', 'navy', 'ayu'];
const TROUPEKIT_THEME_ICONS =
    '<span class="fa-svg troupekit-theme">' +
    '<svg class="ln i-sun" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>' +
    '<svg class="ln i-moon" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>' +
    '</span>';
const TROUPEKIT_PDF_ICON =
    '<span class="fa-svg"><svg viewBox="0 0 384 512"><path d="M64 0C28.7 0 0 28.7 0 64V448c0 35.3 28.7 64 64 64H320c35.3 0 64-28.7 64-64V160H256c-17.7 0-32-14.3-32-32V0H64zM256 0V128H384L256 0zM216 232V334.1l31-31c9.4-9.4 24.6-9.4 33.9 0s9.4 24.6 0 33.9l-72 72c-9.4 9.4-24.6 9.4-33.9 0l-72-72c-9.4-9.4-9.4-24.6 0-33.9s24.6-9.4 33.9 0l31 31V232c0-13.3 10.7-24 24-24s24 10.7 24 24z"/></svg></span>' +
    '<span class="label">PDF</span>';

/**
 * Puts one of the two site themes on the page. Light saves 'rust' under
 * mdbook-theme; dark clears the key, since book.js only accepts saved
 * themes that have a stock menu button and dark is the default anyway.
 * Input: 'rust-dark' or 'rust'. Output: none; html class and label change.
 */
function troupekitApplyTheme(theme, button) {
    console.assert(TROUPEKIT_THEMES.includes(theme), 'unknown theme ' + theme);
    const html = document.documentElement;
    html.classList.remove(...TROUPEKIT_THEMES, ...TROUPEKIT_STOCK_THEMES);
    html.classList.add(theme);
    const next = theme === 'rust' ? 'dark' : 'light';
    button.title = 'Switch to ' + next;
    button.setAttribute('aria-label', 'Switch to ' + next);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta !== null) {
        meta.content = getComputedStyle(html).backgroundColor;
    }
    try {
        if (theme === 'rust') {
            localStorage.setItem('mdbook-theme', theme);
        } else {
            localStorage.removeItem('mdbook-theme');
        }
    } catch (e) {
        console.debug('theme not saved', e);
    }
}

/**
 * Replaces the stock theme menu button with a dark and light switch.
 * rust-dark is the default; any other saved theme falls back to it.
 * Input: none. Output: none; the menu bar button is swapped in place.
 */
function troupekitThemeSwitch() {
    const stock = document.getElementById('mdbook-theme-toggle');
    console.assert(stock !== null, 'mdBook 0.5 theme button');
    const button = stock.cloneNode(false);
    button.removeAttribute('aria-haspopup');
    button.removeAttribute('aria-expanded');
    button.removeAttribute('aria-controls');
    button.innerHTML = TROUPEKIT_THEME_ICONS;
    stock.replaceWith(button);
    let saved = null;
    try {
        saved = localStorage.getItem('mdbook-theme');
    } catch (e) {
        console.debug('theme not read', e);
    }
    troupekitApplyTheme(TROUPEKIT_THEMES.includes(saved) ? saved : 'rust-dark', button);
    button.addEventListener('click', function() {
        const now = document.documentElement.classList.contains('rust') ? 'rust' : 'rust-dark';
        troupekitApplyTheme(now === 'rust' ? 'rust-dark' : 'rust', button);
    });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
        setTimeout(function() {
            const now = document.documentElement.classList.contains('rust') ? 'rust' : 'rust-dark';
            troupekitApplyTheme(now, button);
        }, 0);
    });
}

/**
 * Adds a Download PDF link to the right of the print button. The PDF is
 * troupekit.pdf at the site root, built in CI from print.html.
 * Input: none. Output: none; one link is added to the menu bar.
 */
function troupekitPdfLink() {
    const right = document.querySelector('#mdbook-menu-bar .right-buttons');
    console.assert(right !== null, 'mdBook 0.5 menu bar');
    const link = document.createElement('a');
    link.className = 'troupekit-pdf';
    link.href = path_to_root + 'troupekit.pdf';
    link.title = 'Download PDF';
    link.setAttribute('aria-label', 'Download PDF');
    link.innerHTML = TROUPEKIT_PDF_ICON;
    const print = right.querySelector('a[href$="print.html"]');
    if (print !== null) {
        print.after(link);
    } else {
        right.prepend(link);
    }
}

troupekitThemeSwitch();
troupekitPdfLink();
