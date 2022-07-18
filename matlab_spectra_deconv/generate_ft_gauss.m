function f = ft_gaussian(sigma_cm)
  n = 1000;
  t = 1:n; % matlab starts at 1
  Range = (2000-0.1); % hard coded in python scripts
  T = Range/n;
  freq = fftshift(fftfreq(1000,T));
  ft_gauss = exp(-2*(pi^2)*(freq.^2)*(sigma_cm^2));
  f = ft_gauss;
end