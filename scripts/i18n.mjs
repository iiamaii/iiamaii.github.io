export const languages = ['ko', 'en'];
export const languageRoute = (route, lang) => lang === 'en' ? `/en${route}` : route;

export function localizedConfig(config, lang) {
  const translation = config.translations?.[lang] ?? {};
  return {
    ...config, ...translation,
    profile: { ...config.profile, ...translation.profile },
    philosophy: { ...config.philosophy, ...translation.philosophy },
    hero: { ...config.hero, ...translation.hero },
    logo: config.logo ? { ...config.logo, ...translation.logo } : null,
    topics: (config.topics ?? []).map(topic => ({ ...topic, ...translation.topics?.[topic.id], name: topic.names?.[lang] || translation.topics?.[topic.id]?.name || topic.name }))
  };
}

export const copy = {
  ko: {
    locale: 'ko_KR', home: '홈', skip: '본문으로 건너뛰기', menu: '주 메뉴', profile: '내 프로필', reviews: '논문 리뷰', newTab: '새 탭', language: '언어 선택', koLink: '한글로 보기', enLink: 'Read in English', darkTheme: '다크 모드로 전환', lightTheme: '라이트 모드로 전환',
    profileLink: '내 프로필 보기', collectionTitle: '논문으로 이어지는 질문들.', collectionIntro: '주제별로 읽고, 생각하고, 남긴 기록.', allReviews: '전체 리뷰',
    latestIntro: n => `최근 업데이트한 논문 리뷰 ${n}편을 모았습니다.`, moreReviews: '더보기', updated: '업데이트',
    reviewUi: {
      filterToggle: '필터', searchToggle: '검색', size: '표시 개수', sizeUnit: '개', search: '단어 검색', searchPlaceholder: '제목, 요약, 본문, 저자에서 검색',
      topic: '주제 선택', allTopics: '모든 주제', selectedTopics: '{count}개 선택', clearTopics: '선택 해제', noTopics: '아직 공개된 주제가 없습니다.', topicHint: '선택한 주제 중 하나라도 포함된 리뷰를 표시합니다. 선택하지 않으면 전체를 표시합니다.', from: '업데이트 시작', to: '업데이트 종료', timeHint: '시간은 한국 시간(KST) 기준입니다.',
      sort: '정렬', newest: '최근 업데이트순', oldest: '오래된 업데이트순', newestShort: '최신순', oldestShort: '오래된순', filterApply: '필터 적용', filterReset: '필터 초기화', apply: '검색 적용', searchReset: '검색 초기화', reset: '초기화',
      noResults: '조건에 맞는 리뷰가 없습니다.', noResultsHint: '검색어를 바꾸거나 필터를 초기화해 보세요.',
      range: '전체 {total}편 중 {end}편 표시', zero: '검색 결과 0편', loadMore: '더 불러오기', rangeError: '종료 시간은 시작 시간 이후로 선택해 주세요.', active: '적용 중'
    },
    topicLink: (name, n) => `${name} 리뷰 ${n}편 보기`, moreTopic: '이 주제의 리뷰 더 보기', readingTime: n => `${n}분 읽기`,
    preparing: '리뷰 준비 중', emptyTitle: '첫 논문을 기다리는 자리.', topicIntro: '이 주제의 논문 리뷰를 모읍니다.', emptyIntro: '논문을 등록하면 썸네일과 짧은 소개가 여기에 나타납니다.', noReviews: '아직 등록된 논문이 없습니다.',
    emptyCollectionTitle: '아직 공개된 리뷰가 없습니다.', emptyCollectionIntro: '읽고 생각한 논문들을 이곳에 차근차근 모아갑니다.',
    philosophyQuestion: '어떤 질문을 중요하게 생각하고,\n어떤 태도로 연구하고 싶은가.', philosophyPending: '나의 철학을 담을 문장을 준비하고 있습니다.',
    introPending: '소개 준비 중', aboutMe: '나는 어떤 사람인가.', bioPending: '소개와 연구 관심사를 이곳에 차근차근 채워갑니다.', interests: '관심 있는 질문들.', interestsPending: '연구 관심사를 준비하고 있습니다.', browseReviews: '논문 리뷰 둘러보기',
    reviewHeading: '읽고, 생각하고,\n다시 질문합니다', reviewDescription: '논문의 핵심 아이디어와 나의 생각을 주제별로 모읍니다.', total: n => `총 ${n}편의 리뷰`, reviewList: '논문 리뷰 목록',
    uploaded: '포스트 업로드', paperPublished: '논문 공개', paperAuthors: '저자', allAuthors: '전체 저자 보기', collapseAuthors: '저자 접기', originalPaper: 'ORIGINAL PAPER', paperLink: '논문 원문', toc: '이 글의 목차', related: '같은 주제의 리뷰',
    originalLabel: source => source === 'en' ? 'English 원문' : '한국어 원문',
    originalNotice: source => source === 'en' ? '한국어 번역은 아직 준비되지 않아 영어 원문을 표시합니다.' : '영문 번역은 아직 준비되지 않아 한국어 원문을 표시합니다.',
    profileOriginal: '번역을 준비하고 있어 원문을 표시합니다.',
    notFound: '페이지를 찾을 수 없습니다', notFoundTitle: '아직 없는 기록입니다.', notFoundIntro: '논문 리뷰 목록에서 다른 기록을 찾아보세요.', reviewLink: '논문 리뷰 보기', redirect: '내 프로필로 이동', table: '본문 표'
  },
  en: {
    locale: 'en_US', home: 'home', skip: 'Skip to content', menu: 'Main navigation', profile: 'My profile', reviews: 'Paper reviews', newTab: 'new tab', language: 'Choose language', koLink: '한글로 보기', enLink: 'Read in English', darkTheme: 'Switch to dark mode', lightTheme: 'Switch to light mode',
    profileLink: 'About me', collectionTitle: 'Questions carried by papers.', collectionIntro: 'Reading, thinking, and taking notes by topic.', allReviews: 'All reviews',
    latestIntro: n => `The ${n} most recently updated paper review${n === 1 ? '' : 's'}.`, moreReviews: 'View more', updated: 'Updated',
    reviewUi: {
      filterToggle: 'Filter', searchToggle: 'Search', size: 'Reviews shown', sizeUnit: '', search: 'Word search', searchPlaceholder: 'Search titles, summaries, full text, and authors',
      topic: 'Choose topics', allTopics: 'All topics', selectedTopics: '{count} selected', clearTopics: 'Clear selection', noTopics: 'No public topics yet.', topicHint: 'Reviews matching any selected topic are included. Leave unchecked to show all.', from: 'Updated from', to: 'Updated through', timeHint: 'Times are in Korea Standard Time (KST).',
      sort: 'Sort by', newest: 'Recently updated', oldest: 'Oldest update first', newestShort: 'Newest', oldestShort: 'Oldest', filterApply: 'Apply filters', filterReset: 'Reset filters', apply: 'Apply search', searchReset: 'Reset search', reset: 'Reset',
      noResults: 'No matching reviews.', noResultsHint: 'Try different words or reset the filters.',
      range: 'Showing {end} of {total}', zero: '0 matching reviews', loadMore: 'Load more', rangeError: 'Choose an end time on or after the start time.', active: 'Active'
    },
    topicLink: (name, n) => `View ${name}: ${n} review${n === 1 ? '' : 's'}`, moreTopic: 'More reviews in this topic', readingTime: n => `${n} min read`,
    preparing: 'REVIEW IN PROGRESS', emptyTitle: 'A place for the first paper.', topicIntro: 'Paper reviews in this topic.', emptyIntro: 'New reviews will appear here with a thumbnail and a short introduction.', noReviews: 'No papers added yet.',
    emptyCollectionTitle: 'No public reviews yet.', emptyCollectionIntro: 'A growing collection of papers, reflections, and open questions.',
    philosophyQuestion: 'Which questions matter to me,\nand how do I want to approach research?', philosophyPending: 'A statement of my philosophy is taking shape.',
    introPending: 'Introduction coming soon', aboutMe: 'A little about me.', bioPending: 'A space for my introduction and research interests.', interests: 'Questions I care about.', interestsPending: 'Research interests coming soon.', browseReviews: 'Explore paper reviews',
    reviewHeading: 'Read, reflect,\nand ask again', reviewDescription: 'Key ideas from papers and my reflections, organized by topic.', total: n => `${n} review${n === 1 ? '' : 's'} in total`, reviewList: 'Paper review collection',
    uploaded: 'Post uploaded', paperPublished: 'Paper released', paperAuthors: 'Authors', allAuthors: 'Show all authors', collapseAuthors: 'Collapse authors', originalPaper: 'ORIGINAL PAPER', paperLink: 'Read the paper', toc: 'In this review', related: 'More in this topic',
    originalLabel: source => source === 'ko' ? 'Korean original' : 'English original',
    originalNotice: source => source === 'ko' ? 'An English translation is not available yet. The Korean original is shown below.' : 'A Korean translation is not available yet. The English original is shown below.',
    profileOriginal: 'A translation is in progress. The original text is shown below.',
    notFound: 'Page not found', notFoundTitle: 'This page is still unwritten.', notFoundIntro: 'Explore the paper review collection for more notes.', reviewLink: 'Browse paper reviews', redirect: 'Go to my profile', table: 'Table in this review'
  }
};
