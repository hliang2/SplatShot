// Copy BibTeX
document.addEventListener('DOMContentLoaded', () => {
  const copyBtn = document.querySelector('.copy-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const bibtex = document.querySelector('.bibtex-block code').innerText;
      navigator.clipboard.writeText(bibtex).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 2000);
      });
    });
  }

  // Smooth active nav link
  const sections = document.querySelectorAll('.section[id]');
  const navLinks = document.querySelectorAll('.navbar-links a');
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(a => {
          a.style.color = a.getAttribute('href') === '#' + entry.target.id
            ? 'var(--accent)' : '';
        });
      }
    });
  }, { threshold: 0.4 });
  sections.forEach(s => observer.observe(s));
});
