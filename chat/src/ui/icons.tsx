// Inline stroke icons (no dependency). Styled by the surrounding .panel svg rule.

const svgProps = { viewBox: "0 0 24 24", "aria-hidden": true, focusable: false } as const;

export const PlusIcon = () => (
  <svg {...svgProps}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);

export const CloseIcon = () => (
  <svg {...svgProps}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);

export const WarnIcon = () => (
  <svg {...svgProps}>
    <path d="M12 4l9 16H3z" />
    <path d="M12 10v4M12 17.2v.1" />
  </svg>
);

export const InfoIcon = () => (
  <svg {...svgProps}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5M12 7.8v.1" />
  </svg>
);

export const CheckCircleIcon = () => (
  <svg {...svgProps}>
    <circle cx="12" cy="12" r="9" />
    <path d="M8 12.5l2.8 2.8L16.5 9.5" />
  </svg>
);

export const SearchIcon = () => (
  <svg {...svgProps}>
    <circle cx="11" cy="11" r="6.5" />
    <path d="M20 20l-4.2-4.2" />
  </svg>
);

export const StarIcon = ({ filled = false }: { filled?: boolean }) => (
  <svg {...svgProps}>
    <path
      d="M12 3.8l2.5 5.2 5.7.8-4.1 4 1 5.6-5.1-2.7-5.1 2.7 1-5.6-4.1-4 5.7-.8z"
      fill={filled ? "currentColor" : "none"}
    />
  </svg>
);

export const RefreshIcon = () => (
  <svg {...svgProps}>
    <path d="M20 12a8 8 0 1 1-2.3-5.6" />
    <path d="M20 4v5h-5" />
  </svg>
);

export const ExternalIcon = () => (
  <svg {...svgProps}>
    <path d="M14 4h6v6M20 4l-9 9" />
    <path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />
  </svg>
);
