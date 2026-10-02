/* Plain-English labels, descriptions and families for the model's 35 features (+ their missing-value indicators). */
const F = (label, family, desc, kind) => ({label, family, desc, kind});
const FEATURES = {
 url_length:F('URL length','lexical','Total characters in the URL. Phishing URLs tend to be longer.','int'),
 hostname_length:F('Hostname length','lexical','Characters in the host name (e.g. "www.example.com"). Long hosts often hide extra words or subdomains.','int'),
 path_length:F('Path length','lexical','Characters after the domain. A bare "/" has length 1; long paths are common in phishing kits.','int'),
 num_dots:F('Dots in URL','lexical','Number of "." characters. Many dots usually mean many subdomains.','int'),
 num_hyphens:F('Hyphens in URL','lexical','Hyphens are common in look-alike domains such as "secure-login-bank".','int'),
 num_underscores:F('Underscores','lexical','Number of "_" characters in the URL.','int'),
 num_slashes:F('Slashes','lexical','Number of "/" characters, a rough measure of path depth.','int'),
 num_question_marks:F('Question marks','lexical','Number of "?" characters (query strings).','int'),
 num_equals:F('Equals signs','lexical','Number of "=" characters (query parameters).','int'),
 num_at_symbols:F('"@" symbols','lexical','An "@" can make a browser ignore everything before it, a classic trick.','int'),
 num_percent:F('Percent-encodings','lexical','Number of "%" escapes, sometimes used to disguise characters.','int'),
 num_digits:F('Digits in URL','lexical','Count of digits; random-looking hosts contain many.','int'),
 num_subdomains:F('Subdomains','lexical','Labels before the main domain. Note: "www" counts as one, a known quirk of this model.','int'),
 has_ip_address:F('Host is an IP address','lexical','Legitimate sites almost always use names, not raw IP addresses.','bool'),
 uses_https:F('Uses HTTPS','lexical','Whether the URL starts with https. Now common on phishing too, so weak evidence alone.','bool'),
 has_port:F('Unusual port','lexical','Whether the URL specifies a non-default port.','bool'),
 is_known_shortener:F('URL shortener','lexical','Whether the host is a known link shortener that hides the real destination.','bool'),
 double_slash_in_path:F('"//" inside path','lexical','A second "//" can be used to fake a redirect.','bool'),
 suspicious_tld_in_path:F('Fake TLD in path','lexical','Text like ".com" placed inside the path to look like a domain.','bool'),
 brand_keyword_in_subdomain:F('Brand name in subdomain','lexical','A brand name such as "paypal" used as a subdomain of another site.','bool'),
 domain_age_days:F('Domain age','domain','Days since the domain was registered (WHOIS). New domains are riskier.','days'),
 domain_registration_length_days:F('Registration length','domain','Days between registration and expiry. Short registrations are typical of throwaway sites.','days'),
 dns_resolves:F('DNS resolves','domain','Whether the domain name currently points to a server.','bool'),
 tranco_rank:F('Popularity rank','domain','Rank in the Tranco top-1M sites list. Note: hosting platforms are also highly ranked, a known bias.','rank'),
 num_forms:F('Forms on page','content','Number of HTML forms on the fetched page.','int'),
 form_action_external_ratio:F('Forms posting elsewhere','content','Share of forms that send data to a different domain.','ratio'),
 num_iframes:F('Iframes','content','Embedded frames, sometimes used to overlay fake content.','int'),
 num_external_links:F('External links','content','Links from the page to other domains.','int'),
 num_internal_links:F('Internal links','content','Links to pages on the same site. Real sites have many; phishing pages very few.','int'),
 external_link_ratio:F('External-link share','content','External links as a fraction of all links.','ratio'),
 has_password_field:F('Password field','content','Whether the page asks for a password.','bool'),
 favicon_external:F('Favicon from another domain','content','Whether the site icon is loaded from elsewhere.','bool'),
 num_scripts:F('Scripts','content','Number of script tags on the page.','int'),
 title_hostname_overlap:F('Title matches host','content','Whether the page title mentions the host name.','bool'),
 response_status_code:F('HTTP status','content','Server response code when the page was fetched (200 = OK).','int')
};
function featInfo(name){
  if(FEATURES[name]) return FEATURES[name];
  if(name.endsWith('_was_missing')){
    const p=featInfo(name.slice(0,-12));
    return F('Missing flag: '+p.label,p.family,'Set to 1 when "'+p.label+'" could not be retrieved. The model treats "unavailable" as a signal in itself.','flag');
  }
  return F(name.replace(/_/g,' '),'lexical','','int');
}
function fmtValue(name,v,missing){
  const k=featInfo(name).kind;
  if(k==='flag') return v?'yes (could not be retrieved)':'no (was available)';
  if(missing||v===null||v===undefined) return 'unavailable';
  if(k==='bool') return v?'Yes':'No';
  if(k==='days') return Math.round(v).toLocaleString()+' days (~'+(v/365).toFixed(1)+' yrs)';
  if(k==='rank') return '#'+Math.round(v).toLocaleString();
  if(k==='ratio') return (v*100).toFixed(0)+'%';
  return Number.isInteger(v)?String(v):v.toFixed(2);
}
const FAMILY={lexical:{name:'URL text',color:'#4f46e5'},domain:{name:'Domain lookups',color:'#10b981'},content:{name:'Live page',color:'#f59e0b'}};
