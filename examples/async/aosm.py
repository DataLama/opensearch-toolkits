import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv, find_dotenv

from opensearch_toolkits.core import OpenSearchConfig, AsyncOpenSearchManager

# .env 파일에서 환경 변수 로드
load_dotenv(find_dotenv())

# OpenSearchManager 비동기 방식 사용 예제
async def async_example():
    """AsyncOpenSearchManager 기능을 시연하는 예제"""
    
    print("=== AsyncOpenSearchManager 예제 ===\n")
    
    # 1. 기본 구성으로 관리자 초기화
    config = OpenSearchConfig(
        host=os.environ["OPENSEARCH_HOST"],  # OpenSearch 서버 주소
        port=int(os.environ["OPENSEARCH_PORT"]),  # OpenSearch 서버 포트
        user=os.environ["OPENSEARCH_USER"],  # OpenSearch 사용자 이름
        password=os.environ["OPENSEARCH_PASSWORD"],  # OpenSearch 비밀번호
        use_ssl=os.environ["OPENSEARCH_USE_SSL"].lower() == "true",  # SSL 사용 여부
    )
    
    # 컨텍스트 관리자를 사용하여 자동으로 리소스 정리
    async with AsyncOpenSearchManager(config) as manager:
        # 2. 연결 확인
        print("## 연결 상태 확인")
        connection_status = await manager.check_connection()
        print(f"연결 상태: {'성공' if connection_status else '실패'}")
        print()
        
        # 연결이 성공적이지 않으면 나머지 작업 건너뛰기
        if not connection_status:
            print("OpenSearch 서버에 연결할 수 없습니다. 설정을 확인하세요.")
            return
        
        # 3. 클러스터 정보 가져오기
        print("## 클러스터 정보")
        cluster_info = await manager.get_cluster_info()
        print(f"클러스터 이름: {cluster_info.get('cluster_name')}")
        print(f"버전: {cluster_info.get('version', {}).get('number')}")
        print()
        
        # 4. 클러스터 상태 확인
        print("## 클러스터 상태")
        health = await manager.get_cluster_health()
        print(f"상태: {health.get('status')}")
        print(f"노드 수: {health.get('number_of_nodes')}")
        print(f"데이터 노드 수: {health.get('number_of_data_nodes')}")
        print()
        
        # 5. 인덱스 목록 가져오기
        print("## 인덱스 목록")
        indices = await manager.get_all_indices()
        if indices:
            print(f"인덱스 수: {len(indices)}")
            for idx, index in enumerate(indices[:5], 1):  # 처음 5개만 출력
                print(f"{idx}. {index}")
            
            if len(indices) > 5:
                print(f"... 그 외 {len(indices) - 5}개 인덱스")
        else:
            print("인덱스가 없습니다.")
        print()
        
        # 6. 특정 인덱스 확인 및 작업
        test_index_name = "test-opensearch-index"
        
        # 6.1 인덱스 존재 여부 확인
        index_exists = await manager.index_exists(test_index_name)
        if index_exists:
            print(f"'{test_index_name}' 인덱스가 존재합니다.")
            
            # 6.2 인덱스 통계 확인
            print(f"## '{test_index_name}' 인덱스 통계")
            stats = await manager.get_index_stats(test_index_name)
            print(f"문서 수: {stats.get('doc_count', 0)}")
            print(f"크기: {stats.get('size_bytes', 0)} 바이트")
            print(f"샤드 상태: {stats.get('shards', {})}")
            
            # 6.3 인덱스 매핑 확인
            mappings = await manager.get_mappings(test_index_name)
            print(f"## '{test_index_name}' 인덱스 매핑")
            # 실제 매핑 내용은 크기가 클 수 있으므로 필요한 부분만 출력
            print(f"매핑 필드 수: {len(mappings.get(test_index_name, {}).get('mappings', {}).get('properties', {}))}")
            
            # 6.4 인덱스 설정 업데이트
            print(f"## '{test_index_name}' 인덱스 설정 업데이트")
            settings = {
                "settings": {
                    "index.number_of_replicas": 2,  # 복제본 수 설정
                    "index.refresh_interval": "5s"   # 새로고침 간격 설정
                }
            }
            
            success = await manager.update_index_settings(test_index_name, settings)
            print(f"설정 업데이트 결과: {'성공' if success else '실패'}")
            
            # 6.5 인덱스 복제
            clone_index_name = f"{test_index_name}-clone-async"
            print(f"## '{test_index_name}'를 '{clone_index_name}'로 복제")
            
            # 대상 인덱스가 이미 있는지 확인
            if await manager.index_exists(clone_index_name):
                print(f"'{clone_index_name}' 인덱스가 이미 존재합니다.")
            else:
                success = await manager.clone_index(test_index_name, clone_index_name)
                print(f"인덱스 복제 결과: {'성공' if success else '실패'}")
        else:
            print(f"'{test_index_name}' 인덱스가 존재하지 않습니다.")
        print()
        
        # 7. 샤드 정보 확인
        print("## 샤드 정보")
        shard_info = await manager.get_shard_info()
        print(f"샤드 총 개수: {len(shard_info)}")
        
        # 샤드 상태별 개수 집계
        shard_status = {}
        for shard in shard_info:
            status = shard.get('state', 'unknown')
            shard_status[status] = shard_status.get(status, 0) + 1
        
        print("샤드 상태 분포:")
        for status, count in shard_status.items():
            print(f"- {status}: {count}개")
        print()
        
        # 8. 클러스터 상세 통계
        print("## 클러스터 상세 통계")
        cluster_stats = await manager.get_cluster_stats()
        print(f"인덱스 수: {cluster_stats.get('indices', {}).get('count')}")
        print(f"샤드 수: {cluster_stats.get('indices', {}).get('shards', {}).get('total')}")
        print(f"문서 수: {cluster_stats.get('indices', {}).get('docs', {}).get('count')}")
        print(f"저장 크기: {cluster_stats.get('indices', {}).get('store', {}).get('size_in_bytes')} 바이트")
        print()
        
        # 9. 병렬 작업 수행 예제
        print("## 병렬 작업 수행 (비동기의 장점)")
        # 여러 인덱스의 통계를 동시에 가져오기
        if len(indices) >= 3:
            sample_indices = indices[:3]  # 예제로 3개만 선택
            
            print(f"3개의 인덱스 통계를 병렬로 가져오기:")
            # asyncio.gather로 여러 작업을 병렬로 실행
            stats_tasks = [manager.get_index_stats(idx) for idx in sample_indices]
            all_stats = await asyncio.gather(*stats_tasks)
            
            for i, stats in enumerate(all_stats):
                print(f"{sample_indices[i]}: {stats.get('doc_count', 0)}개 문서")
        else:
            print("인덱스가 충분하지 않습니다. 병렬 작업 예제를 건너뜁니다.")
        print()
    
    # 컨텍스트 관리자가 끝나면 자동으로 리소스가 정리됨
    print("=== 비동기 방식 예제 완료 (모든 리소스가 자동으로 정리됨) ===")

# 비동기 함수 실행
if __name__ == "__main__":
    asyncio.run(async_example())