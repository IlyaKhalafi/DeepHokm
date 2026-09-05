"""Policy wiring for MaskablePPO over the Hokm transformer extractor.

:class:`HokmMaskablePolicy` binds sb3-contrib's
:class:`~sb3_contrib.MaskableActorCriticPolicy` to
:class:`~deephokm.nn.extractor.HokmTransformerExtractor`. The orthogonal
initialization gains this project specifies — 0.01 for the policy (action)
head, 1.0 for the value head — are exactly the gains the parent class applies
in ``_build``; this subclass pins them by test rather than by re-implementation.

Logit masking is handled entirely by ``MaskablePPO`` — nothing here
re-implements it.
"""

from __future__ import annotations

from typing import Any

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from deephokm.nn.extractor import HokmTransformerExtractor

POLICY_HEAD_GAIN = 0.01
VALUE_HEAD_GAIN = 1.0


class HokmMaskablePolicy(MaskableActorCriticPolicy):
    """Maskable actor-critic policy over the Hokm transformer.

    The feature extractor defaults to :class:`HokmTransformerExtractor` and
    ``net_arch`` defaults to an empty list so the transformer's pooled output
    feeds the policy and value heads directly. Orthogonal initialization runs
    with the parent's gain table: sqrt(2) for the extractor, 0.01 for the
    action net, 1.0 for the value net.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        lr_schedule: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Create the policy.

        Accepts the standard ``MaskableActorCriticPolicy`` arguments;
        ``features_extractor_class`` and ``net_arch`` are defaulted as
        described in the class docstring and can still be overridden.
        """
        kwargs.setdefault("features_extractor_class", HokmTransformerExtractor)
        kwargs.setdefault("net_arch", [])
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
