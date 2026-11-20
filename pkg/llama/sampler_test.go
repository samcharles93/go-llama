package llama

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// The stage order is llama.cpp's own default order. Getting it wrong changes
// what the temperature means, so it is pinned here rather than inferred from
// sampled tokens.
func TestNewSamplerFromOptionsStageOrder(t *testing.T) {
	opts := CompletionOptions{
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.95,
		MinP:             0.05,
		TypicalP:         0.2,
		TopNSigma:        1.5,
		XTCProbability:   0.5,
		XTCThreshold:     0.1,
		RepeatPenalty:    1.1,
		RepeatLastN:      64,
		PresencePenalty:  0.2,
		FrequencyPenalty: 0.1,
		Seed:             1234,
	}
	s, err := newSamplerFromOptions(nil, opts)
	if err != nil {
		t.Fatalf("newSamplerFromOptions: %v", err)
	}
	defer s.Close()

	want := []string{"penalties", "top-n-sigma", "top-k", "typical", "top-p", "min-p", "xtc", "temp", "dist"}
	if got := s.Stages(); !slices.Equal(got, want) {
		t.Fatalf("stages = %v, want %v", got, want)
	}
}

func TestNewSamplerFromOptionsMirostatReplacesPicker(t *testing.T) {
	opts := CompletionOptions{Temperature: 0.7, MirostatTau: 5, MirostatEta: 0.1, Seed: 7}
	s, err := newSamplerFromOptions(nil, opts)
	if err != nil {
		t.Fatalf("newSamplerFromOptions: %v", err)
	}
	defer s.Close()

	want := []string{"temp", "mirostat-v2"}
	if got := s.Stages(); !slices.Equal(got, want) {
		t.Fatalf("stages = %v, want %v", got, want)
	}
}

func TestSamplerDefaultsToGreedy(t *testing.T) {
	cases := map[string]CompletionOptions{
		"zero value":         {},
		"greedy helper":      GreedyCompletionOptions(8),
		"temperature zero":   {Temperature: 0, TopK: 40, TopP: 0.9},
		"temperature negatv": {Temperature: -1, TopK: 40},
	}
	for name, opts := range cases {
		t.Run(name, func(t *testing.T) {
			s, err := newSamplerFromOptions(nil, opts)
			if err != nil {
				t.Fatalf("newSamplerFromOptions: %v", err)
			}
			defer s.Close()
			if got := s.Stages(); !slices.Equal(got, []string{"greedy"}) {
				t.Fatalf("stages = %v, want greedy only", got)
			}
		})
	}
}

func TestSamplerGreedyKeepsPenalties(t *testing.T) {
	s, err := newSamplerFromOptions(nil, CompletionOptions{Temperature: 0, RepeatPenalty: 1.2})
	if err != nil {
		t.Fatalf("newSamplerFromOptions: %v", err)
	}
	defer s.Close()
	if got := s.Stages(); !slices.Equal(got, []string{"penalties", "greedy"}) {
		t.Fatalf("stages = %v, want penalties then greedy", got)
	}
}

func TestSamplerDisabledStagesAreAbsent(t *testing.T) {
	// A temperature alone leaves exactly the temperature stage and the picker.
	s, err := newSamplerFromOptions(nil, CompletionOptions{Temperature: 0.5})
	if err != nil {
		t.Fatalf("newSamplerFromOptions: %v", err)
	}
	defer s.Close()
	if got := s.Stages(); !slices.Equal(got, []string{"temp", "dist"}) {
		t.Fatalf("stages = %v, want temp then dist", got)
	}
}

func TestSamplerXTCThresholdDisablesStage(t *testing.T) {
	opts := CompletionOptions{Temperature: 0.5, XTCProbability: 0.5, XTCThreshold: 0.6}
	if opts.xtcEnabled() {
		t.Fatal("xtcEnabled reports true above llama.cpp's 0.5 threshold")
	}
	s, err := newSamplerFromOptions(nil, opts)
	if err != nil {
		t.Fatalf("newSamplerFromOptions: %v", err)
	}
	defer s.Close()
	if slices.Contains(s.Stages(), "xtc") {
		t.Fatalf("stages = %v, want no xtc stage", s.Stages())
	}
}

func TestSamplerDRYNeedsVocabulary(t *testing.T) {
	_, err := newSamplerFromOptions(nil, CompletionOptions{Temperature: 0.7, DryMultiplier: 0.8})
	if err == nil {
		t.Fatal("DRY without a vocabulary returned no error")
	}
	if !errors.Is(err, ErrDecode) {
		t.Fatalf("error %v does not wrap ErrDecode", err)
	}
}

func TestSamplerGrammarNeedsVocabulary(t *testing.T) {
	_, err := newSamplerFromOptions(nil, CompletionOptions{Grammar: `root ::= "a"`})
	if err == nil {
		t.Fatal("grammar without a vocabulary returned no error")
	}
	if !errors.Is(err, ErrGrammar) {
		t.Fatalf("error %v does not wrap ErrGrammar", err)
	}
}

func TestNewSamplerFromOptionsRejectsInvalidOptions(t *testing.T) {
	cases := map[string]CompletionOptions{
		"nan temperature":       {Temperature: float32(math.NaN())},
		"infinite temperature":  {Temperature: float32(math.Inf(1))},
		"nan top p":             {Temperature: 0.5, TopP: float32(math.NaN())},
		"infinite top p":        {Temperature: 0.5, TopP: float32(math.Inf(-1))},
		"nan min p":             {Temperature: 0.5, MinP: float32(math.NaN())},
		"nan typical p":         {Temperature: 0.5, TypicalP: float32(math.NaN())},
		"nan top n sigma":       {Temperature: 0.5, TopNSigma: float32(math.NaN())},
		"nan xtc probability":   {Temperature: 0.5, XTCProbability: float32(math.NaN())},
		"xtc probability above": {Temperature: 0.5, XTCProbability: 1.5},
		"xtc threshold below":   {Temperature: 0.5, XTCThreshold: -0.1},
		"negative top n sigma":  {Temperature: 0.5, TopNSigma: -1},
		"negative top k":        {Temperature: 0.5, TopK: -1},
		"top p above one":       {Temperature: 0.5, TopP: 1.5},
		"min p below zero":      {Temperature: 0.5, MinP: -0.1},
		"negative repeat":       {Temperature: 0.5, RepeatPenalty: -0.5},
		"repeat window below":   {Temperature: 0.5, RepeatLastN: -2},
		"nan repeat penalty":    {Temperature: 0.5, RepeatPenalty: float32(math.NaN())},
		"nan dry multiplier":    {Temperature: 0.5, DryMultiplier: float32(math.NaN())},
		"dry base below one":    {Temperature: 0.5, DryMultiplier: 0.8, DryBase: 0.5},
		"negative dry":          {Temperature: 0.5, DryMultiplier: -0.8},
		"negative mirostat tau": {Temperature: 0.5, MirostatTau: -5, MirostatEta: 0.1},
		"dry allowed negative":  {Temperature: 0.5, DryMultiplier: 0.8, DryAllowedLength: -1},
		"dry window below":      {Temperature: 0.5, DryMultiplier: 0.8, DryPenaltyLastN: -2},
		"empty dry breaker":     {Temperature: 0.5, DryMultiplier: 0.8, DrySequenceBreakers: []string{""}},
		"tau without eta":       {Temperature: 0.5, MirostatTau: 5},
		"eta without tau":       {Temperature: 0.5, MirostatEta: 0.1},
		"empty stop":            {Temperature: 0.5, Stop: []string{""}},
	}
	for name, opts := range cases {
		t.Run(name, func(t *testing.T) {
			s, err := newSamplerFromOptions(nil, opts)
			if err == nil {
				s.Close()
				t.Fatal("invalid options built a chain")
			}
			if s != nil {
				t.Fatalf("failed build returned a sampler: %v", s.Stages())
			}
			if !errors.Is(err, ErrDecode) {
				t.Fatalf("error %v does not wrap ErrDecode", err)
			}
		})
	}
}

func TestClampTopK(t *testing.T) {
	cases := []struct {
		name      string
		k         int32
		vocabSize int
		want      int32
	}{
		{"above vocabulary", 500000, 32000, 32000},
		{"inside vocabulary", 40, 32000, 40},
		{"equal to vocabulary", 32000, 32000, 32000},
		{"unknown vocabulary", 500000, 0, 500000},
		{"disabled", 0, 32000, 0},
		{"negative passes through", -1, 32000, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := clampTopK(tc.k, tc.vocabSize); got != tc.want {
				t.Fatalf("clampTopK(%d, %d) = %d, want %d", tc.k, tc.vocabSize, got, tc.want)
			}
		})
	}
}

func TestRepeatLastNResolution(t *testing.T) {
	cases := map[int32]int32{0: 64, -1: -1, 128: 128}
	for in, want := range cases {
		if got := (CompletionOptions{RepeatLastN: in}).repeatLastN(); got != want {
			t.Errorf("RepeatLastN %d resolved to %d, want %d", in, got, want)
		}
	}
}

func TestDRYDefaults(t *testing.T) {
	opts := CompletionOptions{DryMultiplier: 0.8}
	if !opts.dryEnabled() {
		t.Fatal("DryMultiplier does not enable DRY")
	}
	if got := opts.dryBase(); got != 1.75 {
		t.Errorf("dryBase = %v, want llama.cpp's default 1.75", got)
	}
	if got := opts.dryAllowedLength(); got != 2 {
		t.Errorf("dryAllowedLength = %d, want llama.cpp's default 2", got)
	}
	if (CompletionOptions{}).dryEnabled() {
		t.Error("zero options enable DRY")
	}
}

func TestSeedRandomIsLLAMADefault(t *testing.T) {
	if SeedRandom != 0xFFFFFFFF {
		t.Fatalf("SeedRandom = %#x, want llama.cpp's LLAMA_DEFAULT_SEED", SeedRandom)
	}
	if got := DefaultCompletionOptions().Seed; got != SeedRandom {
		t.Fatalf("DefaultCompletionOptions().Seed = %#x, want SeedRandom", got)
	}
}

// The documented order only holds if the chain has one stage per enabled
// option, so count them rather than just checking membership.
func TestSamplerStageCount(t *testing.T) {
	opts := CompletionOptions{Temperature: 0.8, TopK: 40, TopP: 0.9, Seed: 1}
	s, err := newSamplerFromOptions(nil, opts)
	if err != nil {
		t.Fatalf("newSamplerFromOptions: %v", err)
	}
	defer s.Close()
	if got := len(s.Stages()); got != 4 {
		t.Fatalf("chain has %d stages (%v), want 4", got, s.Stages())
	}
}

func TestSamplerNilSafety(t *testing.T) {
	var s *Sampler
	if got := s.Stages(); got != nil {
		t.Fatalf("nil sampler Stages = %v, want nil", got)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("nil sampler Close = %v, want nil", err)
	}

	chain := NewSamplerChain()
	chain.AddGreedy()
	if err := chain.Close(); err != nil {
		t.Fatalf("Close = %v", err)
	}
	if err := chain.Close(); err != nil {
		t.Fatalf("second Close = %v", err)
	}
	if got := chain.Stages(); got != nil {
		t.Fatalf("closed chain Stages = %v, want nil", got)
	}
}
