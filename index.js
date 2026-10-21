/**
 * dsh-music-analyze — 让模型「看见」音乐
 *
 * 注册一个工具 `music_analyze`：把音频/视频交给 python/music_analyze.py，
 * 得到四联图（波形 / 梅尔频谱 / 色度 / 段落能量）+ 文字指标，
 * 模型看图看数字，就能和人讨论这首曲子。
 *
 * 为什么走子进程而不是纯 JS：
 *   librosa + matplotlib 的频谱、色度、节拍检测，JS 生态里没有等价物。
 */
import { execFile } from 'node:child_process'
import { existsSync, realpathSync } from 'node:fs'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

/**
 * 插件真实目录。
 * pnpm 把本地包软链进 profile 的 node_modules，import.meta.url 会是**软链路径**，
 * 那样向上找 `_tools/venv` 就找不到 —— 所以先 realpath 解析回真实位置。
 */
const HERE = (() => {
  const url = fileURLToPath(import.meta.url)
  try {
    return dirname(realpathSync(url))
  } catch {
    return dirname(url)
  }
})()
const SCRIPT = join(HERE, 'python', 'music_analyze.py')

/** 向上若干级，把常见的 venv 布局都列出来（插件自己带 venv 的结构也能命中） */
function venvCandidates() {
  const out = []
  const isWin = process.platform === 'win32'
  for (const base of [HERE, process.cwd()]) {
    let root = base
    for (let up = 0; up < 5; up++) {
      root = up === 0 ? base : dirname(root)
      for (const v of [join(root, '_tools', 'venv'), join(root, '.venv'), join(root, 'venv')]) {
        out.push(isWin ? join(v, 'Scripts', 'python.exe') : join(v, 'bin', 'python'))
      }
    }
  }
  return out.filter((p) => {
    try {
      return existsSync(p)
    } catch {
      return false
    }
  })
}

/** 依次尝试的 Python 解释器 */
function pythonCandidates(config) {
  const list = []
  if (config && typeof config.pythonPath === 'string' && config.pythonPath.trim()) {
    list.push(config.pythonPath.trim())
  }
  if (typeof process.env.MUSIC_ANALYZE_PYTHON === 'string' && process.env.MUSIC_ANALYZE_PYTHON) {
    list.push(process.env.MUSIC_ANALYZE_PYTHON)
  }
  // 插件旁边 / 工作区里的 venv（最常见的情况）
  list.push(...venvCandidates())
  if (process.platform === 'win32') {
    list.push('python', 'python3', 'py')
  } else {
    list.push('python3', 'python')
  }
  return [...new Set(list)]
}

/** 跑一次 python -c "import librosa"，确认依赖在 */
function probePython(exe, timeoutMs = 20000) {
  return new Promise((done) => {
    execFile(
      exe,
      ['-c', 'import librosa, matplotlib, soundfile; print("ok")'],
      { timeout: timeoutMs, windowsHide: true },
      (err, stdout) => done(!err && String(stdout).includes('ok')),
    )
  })
}

/** 找一个能跑 librosa 的 Python */
async function resolvePython(config) {
  const tried = []
  for (const exe of pythonCandidates(config)) {
    tried.push(exe)
    if (await probePython(exe)) return exe
  }
  throw new Error(
    `找不到装了 librosa 的 Python。试过：${tried.join(', ')}。\n` +
      `请先装依赖：\n` +
      `  python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ librosa soundfile matplotlib\n` +
      `或在插件配置里把 pythonPath 指向你的解释器（比如某个 venv 的 python.exe）。`,
  )
}

/** 执行分析脚本，解析最后一行 JSON */
function runScript(python, args, timeoutMs) {
  return new Promise((done, fail) => {
    execFile(
      python,
      [SCRIPT, ...args, '--json'],
      {
        timeout: timeoutMs,
        windowsHide: true,
        maxBuffer: 8 * 1024 * 1024,
        // Windows 上 Python 的 stdout 默认 GBK，这里强制 UTF-8（脚本里也 reconfigure 了一次，双保险）
        env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
      },
      (err, stdout, stderr) => {
        if (err && !String(stdout).includes('{')) {
          const tail = String(stderr || err.message || '').trim().slice(-1200)
          fail(new Error(`分析失败：${tail}`))
          return
        }
        const lines = String(stdout || '').trim().split(/\r?\n/)
        let payload = null
        for (let i = lines.length - 1; i >= 0; i--) {
          const line = lines[i].trim()
          if (!line.startsWith('{')) continue
          try {
            payload = JSON.parse(line)
            break
          } catch {
            /* 继续往前找 */
          }
        }
        if (!payload) {
          const tail = String(stderr || stdout || '').trim().slice(-800)
          fail(new Error(`脚本没有输出可解析的 JSON：${tail}`))
          return
        }
        if (payload.ok === false) {
          fail(new Error(String(payload.error || '分析失败')))
          return
        }
        done(payload)
      },
    )
  })
}

function createTool(config) {
  const cfg = config && typeof config === 'object' ? config : {}

  return {
    name: 'music_analyze',
    description:
      '把音频或视频文件「画」成四张图并给出文字指标，用来和人讨论音乐。' +
      '产物：波形图（节奏/强弱/段落）、梅尔频谱（频率分布）、色度图（12 音高类别，看调性和和弦走向）、段落能量曲线（前奏/主歌/副歌分界），' +
      '外加 BPM、主调、频段能量占比、最响和最安静的时刻。' +
      '**注意**：这些是「音乐的形态」，不是「好不好听」——能不能打动人是听者的判断，图只能告诉你它长什么样。不要把形态描述当成感受。' +
      '分析完请用 read_image 打开返回的 imagePath 自己看一遍，再结合 report 和用户讨论。' +
      '**BPM 用三个独立估计器交叉验证**（动态规划节拍跟踪 / onset 自相关 / 循环节拍图），' +
      '再用「奇偶拍强度比」裁决是否把八分音符误当成拍点；bpmDetail 里有三个原始值、候选和判定来源，' +
      '速度有歧义时把这些一起告诉用户，不要把 bpm 当唯一事实。' +
      '支持 mp3/wav/flac/m4a/mp4 等（内部用 ffmpeg 统一转 22050Hz 单声道）。',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: '音频或视频文件路径（建议绝对路径；相对路径按宿主进程目录解析）' },
        start: { type: 'number', description: '只分析从第几秒开始（可选）' },
        end: { type: 'number', description: '分析到第几秒为止（可选）' },
        outDir: { type: 'string', description: '产物目录；留空 = 音频同级的 music/ 子目录' },
        label: { type: 'string', description: '给这段音频起个名字，写进图标题（可选）' },
      },
      required: ['path'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          ok: { type: 'boolean' },
          source: { type: 'string' },
          imagePath: { type: 'string' },
          reportPath: { type: 'string' },
          report: { type: 'string' },
          duration: { type: 'number' },
          bpm: { type: 'number' },
          bpmReliable: { type: 'boolean' },
          bpmNote: { type: 'string' },
          bpmDetail: { type: 'object', description: '三估计器原始值、奇偶强度比、候选 BPM、判定来源' },
          key: { type: 'string' },
          bands: { type: 'object' },
          loudest: { type: 'array' },
          quietest: { type: 'array' },
        },
        description: '图表路径与音乐指标；report 是给人看的文字版',
      },
      render: (_args, value) => {
        if (!value || value.ok === false) {
          return [{ type: 'text', text: String((value && value.error) || '分析失败') }]
        }
        const lines = []
        lines.push(`🎵 ${value.label || value.source || ''}`)
        lines.push('')
        lines.push(String(value.report || ''))
        lines.push('')
        lines.push(`图：${value.imagePath}`)
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },

    async execute(args) {
      const src = String((args && args.path) || '').trim()
      if (!src) throw new Error('path 不能为空')
      const abs = isAbsolute(src) ? src : resolve(process.cwd(), src)
      if (!existsSync(abs)) throw new Error(`文件不存在：${abs}`)
      if (!existsSync(SCRIPT)) throw new Error(`插件自带的分析脚本缺失：${SCRIPT}`)

      const python = await resolvePython(cfg)

      const pyArgs = [abs]
      if (args.outDir) pyArgs.push('--out', String(args.outDir))
      if (Number.isFinite(Number(args.start))) pyArgs.push('--start', String(Number(args.start)))
      if (Number.isFinite(Number(args.end))) pyArgs.push('--end', String(Number(args.end)))
      if (args.label) pyArgs.push('--label', String(args.label))

      const timeoutMs = Number(cfg.timeoutMs) > 0 ? Number(cfg.timeoutMs) : 600000
      const data = await runScript(python, pyArgs, timeoutMs)

      return {
        ok: true,
        source: abs,
        label: args.label || '',
        imagePath: data.imagePath,
        reportPath: data.reportPath,
        report: data.report,
        duration: data.duration,
        bpm: data.bpm,
        bpmReliable: data.bpmReliable,
        bpmNote: data.bpmNote || '',
        key: data.key,
        bands: data.bands,
        loudest: data.loudest,
        quietest: data.quietest,
      }
    },
  }
}

export const inject = ['tools']

export function apply(ctx, config) {
  return ctx.tools.register(createTool(config))
}
