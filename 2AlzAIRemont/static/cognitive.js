document.addEventListener('DOMContentLoaded', () => {
  // --------- Step navigation & validation ---------
  const tabs = [...document.querySelectorAll('#stepTabs .nav-link')];
  const steps = [...document.querySelectorAll('.step')];

  function showStep(n) {
    steps.forEach((s, i) => s.classList.toggle('d-none', i !== n - 1));
    tabs.forEach((t, i) => t.classList.toggle('active', i === n - 1));
  }

  // Хранилище ответов Этапа 1
  let testResults = {};

  // Переходы между шагами
  [...document.querySelectorAll('.next-step')].forEach(btn => btn.addEventListener('click', () => {
    const next = parseInt(btn.dataset.next);

    if (next === 2) {
      const f = document.getElementById('mmseForm');
      if (!f.checkValidity()) { 
        f.classList.add('was-validated');
        return; 
      }
      // Сохраняем ответы 1-го этапа
      testResults.mmseAnswers = {
        q1: document.getElementById('mmseQ1').value.toLowerCase().trim(),
        q2: document.getElementById('mmseQ2').value.toLowerCase().trim(),
        q3: document.getElementById('mmseQ3').value.toLowerCase().trim(),
        q4: document.getElementById('mmseQ4').value.toLowerCase().trim(),
        q5: document.getElementById('mmseQ5').value.toLowerCase().trim(),
        q6: document.getElementById('mmseQ6').value.toLowerCase().trim(),
        q7: document.getElementById('mmseQ7').value.toLowerCase().trim()
      };
    }

    if (next === 3) {
      const file = document.getElementById('clockImg');
      if (!file.files.length) {
        document.getElementById('clockErr').style.display = 'block';
        return;
      } else {
        document.getElementById('clockErr').style.display = 'none';
      }
      if (!gameStarted) initGame();
    }

    showStep(next);
  }));

  // --------- Memory game logic ---------
  let symbols = ['🍏', '🧠', '🕰️', '💡', '🔑', '📚'];
  let grid = document.getElementById('gameGrid');
  let moveCounter = document.getElementById('moveCount');
  let matchedDisplay = document.getElementById('matched');
  let finishBtn = document.getElementById('finishTest');
  let restartBtn = document.getElementById('restartGame');

  let first = null, lock = false, moves = 0, matchedPairs = 0, gameStarted = false;

  function shuffle(arr) { return arr.sort(() => Math.random() - 0.5); }

  function initGame() {
    gameStarted = true;
    moves = 0;
    matchedPairs = 0;
    moveCounter.textContent = '0';
    matchedDisplay.textContent = '0/6';
    finishBtn.disabled = true;
    grid.innerHTML = '';
    let cards = shuffle([...symbols, ...symbols]);
    cards.forEach(sym => {
      let btn = document.createElement('button');
      btn.className = 'btn btn-outline-dark';
      btn.textContent = '❓';
      btn.dataset.sym = sym;
      btn.onclick = cardClick;
      grid.appendChild(btn);
    });
  }

  function cardClick() {
    if (lock || this.classList.contains('matched')) return;
    this.textContent = this.dataset.sym;
    this.disabled = true;
    if (!first) { first = this; return; }
    moves++;
    moveCounter.textContent = moves;
    if (this.dataset.sym === first.dataset.sym) {
      this.classList.add('matched');
      first.classList.add('matched');
      matchedPairs++;
      matchedDisplay.textContent = `${matchedPairs}/6`;
      first = null;
      if (matchedPairs === 6) { finishBtn.disabled = false; }
    } else {
      lock = true;
      setTimeout(() => {
        this.textContent = '❓';
        first.textContent = '❓';
        this.disabled = false;
        first.disabled = false;
        first = null;
        lock = false;
      }, 800);
    }
  }

  restartBtn?.addEventListener('click', () => initGame());

  // --------- (Опционально) ИИ-комментарий через Gemini ---------
  const GEMINI_API_KEY = ''; // <- вставь ключ, если хочешь ИИ-комментарий
  async function geminiAPI(prompt) {
    if (!GEMINI_API_KEY) return null;
    const url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent';

    const response = await fetch(`${url}?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return data?.candidates?.[0]?.content?.parts?.[0]?.text || '';
  }

  function toPlainHtml(str = '') {
    let s = str.replace(/[*_#`>]+/g, '').replace(/\n{2,}/g, '\n');
    const lines = s.split('\n').map(l => l.trim()).filter(Boolean).slice(0, 8);
    if (!lines.length) return '';
    return `<ul class="mb-0">${lines.map(l => `<li>${l}</li>`).join('')}</ul>`;
  }

  // --------- Мини-консультация (локальная логика) ---------
  function miniAdvice({ total, maxTotal, mmse, clock, memory, moves }) {
    // Категория
    let level = 'Норма';
    if (total <= 4) level = 'Выраженное снижение';
    else if (total <= 7) level = 'Умеренное снижение';

    // Причины по блокам
    const reasons = [];
    if (mmse >= 6) reasons.push('ориентация и базовые когнитивные навыки в пределах нормы по MMSE/MoCA');
    else if (mmse >= 4) reasons.push('по MMSE/MoCA есть отдельные неточности (речь, счёт или абстракции)');
    else reasons.push('низкие баллы по MMSE/MoCA (ориентация, счёт, абстракции требуют внимания)');

    if (clock === 1) reasons.push('задание «Часы» выполнено (основные исполнительные функции сохранены)');
    else reasons.push('ошибки в задании «Часы» (планирование/практические навыки)');

    if (memory === 3 && moves <= 12) reasons.push('память и внимание хорошие (Memory: максимум баллов, мало ходов)');
    else if (memory >= 2) reasons.push('память/внимание средние (Memory в пределах нормы, но ходов больше среднего)');
    else reasons.push('память/внимание снижены (сложно удерживать пары/много ходов)');

    // Что дальше
    let next = '';
    if (total >= 9) {
      next = 'Поддерживайте режим: когнитивные упражнения 10–15 мин/день (чтение, головоломки), умеренная физическая активность, сон 7–8 ч, контроль стресса.';
    } else if (total >= 7) {
      next = 'Рекомендуется тренировать внимание и счёт (короткие устные вычисления, игры на память) 3–4 раза в неделю и повторить тест через 2–4 недели.';
    } else {
      next = 'Рекомендуется обратиться к неврологу/психиатру для очной оценки и исключения медицинских причин (анализы, МРТ, очные когнитивные шкалы).';
    }

    return {
      level,
      reasons,
      next
    };
  }

  // --------- Завершение теста: короткий отчёт + мини-консультация ---------
  document.getElementById('finishTest')?.addEventListener('click', async () => {
    const resultDiv = document.getElementById('cogResult');
    resultDiv.classList.remove('d-none');
    resultDiv.className = 'alert alert-info mt-4';
    resultDiv.textContent = '⏳ Обработка результатов…';

    // Оценка Этапа 1 (MMSE/MoCA)
    let mmseScore = 0;
    const maxMmseScore = 7;
    const today = new Date().toLocaleString('ru-RU', { weekday: 'long' }).toLowerCase();
    if (testResults.mmseAnswers?.q1 === today) mmseScore++;
    if (testResults.mmseAnswers?.q2 === 'казахстан') mmseScore++;
    if (testResults.mmseAnswers?.q3 === '675') mmseScore++;
    if (testResults.mmseAnswers?.q4?.includes('фрукты') || testResults.mmseAnswers?.q4?.includes('круглые')) mmseScore++;
    if (testResults.mmseAnswers?.q5 === 'рим') mmseScore++;
    if (testResults.mmseAnswers?.q6?.includes('пес') || testResults.mmseAnswers?.q6?.includes('светило')) mmseScore++;
    if (testResults.mmseAnswers?.q7 === '189 178 167') mmseScore++;

    const clockScore = 1;       // факт загрузки
    const maxClockScore = 1;

    const memoryScore = matchedPairs === 6 ? (moves <= 12 ? 3 : moves <= 18 ? 2 : 1) : 0;
    const maxMemoryScore = 3;

    const totalScore = mmseScore + clockScore + memoryScore;
    const maxTotalScore = maxMmseScore + maxClockScore + maxMemoryScore; // 11

    // Мини-консультация
    const advice = miniAdvice({
      total: totalScore,
      maxTotal: maxTotalScore,
      mmse: mmseScore,
      clock: clockScore,
      memory: memoryScore,
      moves
    });

    // КОМПАКТНЫЙ HTML-ОТЧЁТ + мини-консультация
    const compactHtml = `
      <div class="result-compact">
        <h4 class="mb-2">Итог: ${totalScore} из ${maxTotalScore} — ${advice.level}</h4>
        <ul>
          <li><strong>Этап 1 (MMSE/MoCA):</strong> ${mmseScore}/7</li>
          <li><strong>Этап 2 (Часы):</strong> ${clockScore}/1</li>
          <li><strong>Этап 3 (Memory):</strong> ${memoryScore}/3, ходов: ${moves}</li>
        </ul>

        <div class="mt-3">
          <strong>Почему такой вывод:</strong>
          <ul class="mb-2">
            ${advice.reasons.map(r => `<li>${r}</li>`).join('')}
          </ul>
          <strong>Что дальше:</strong>
          <p class="mb-0">${advice.next}</p>
        </div>

        <p class="text-muted mt-3 mb-2">Тест является скрининговым и не заменяет консультацию специалиста.</p>
        <div id="aiBox" class="mt-2"></div>
      </div>
    `;

    resultDiv.className = 'alert alert-success mt-4';
    resultDiv.innerHTML = compactHtml;

    // ===== ИИ-комментарий (короткий) — опционально =====
    const aiBox = document.getElementById('aiBox');
    if (!GEMINI_API_KEY) return;

    try {
      aiBox.innerHTML = `<div class="small text-muted">Запрашиваем краткий ИИ-комментарий…</div>`;

      const prompt = `
Сделай очень краткий комментарий по результатам когнитивного скрининга, без Markdown и без длинных текстов (до 5 коротких пунктов). 
Дай поясняющие формулировки и практичный совет. 
Данные:
- Итог: ${totalScore} из ${maxTotalScore}
- Этап 1 (MMSE/MoCA): ${mmseScore}/7
- Этап 2 (Часы): ${clockScore}/1
- Этап 3 (Memory): ${memoryScore}/3, ходов: ${moves}
Если балл ≤ 7 — обязательно рекомендуй очную консультацию врача.
      `.trim();

      const aiText = await geminiAPI(prompt);
      const htmlList = toPlainHtml(aiText);
      aiBox.innerHTML = htmlList
        ? `<details class="mt-2"><summary class="text-primary">Показать краткий ИИ-комментарий</summary><div class="mt-2 small">${htmlList}</div></details>`
        : '';
    } catch {
      aiBox.innerHTML = '';
    }
  });

  // init game once step3 reached via URL hash (optional)
  if (location.hash === "#memory" && grid) initGame();
});
