/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:23:18 GMT 2023
 */

package org.apache.commons.collections4.map;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import org.apache.commons.collections4.Factory;
import org.apache.commons.collections4.functors.ConstantFactory;
import org.apache.commons.collections4.functors.ExceptionFactory;
import org.apache.commons.collections4.map.MultiValueMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MultiValueMap_ESTest extends MultiValueMap_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HashMap<Collection<Object>, Object> hashMap0 = new HashMap<Collection<Object>, Object>();
      LinkedList<HashMap<Collection<Object>, String>> linkedList0 = new LinkedList<HashMap<Collection<Object>, String>>();
      Factory<Collection<HashMap<Collection<Object>, String>>> factory0 = ConstantFactory.constantFactory((Collection<HashMap<Collection<Object>, String>>) linkedList0);
      MultiValueMap<Collection<Object>, HashMap<Collection<Object>, String>> multiValueMap0 = MultiValueMap.multiValueMap((Map<Collection<Object>, ? super Collection<HashMap<Collection<Object>, String>>>) hashMap0, factory0);
      MultiValueMap<Integer, String> multiValueMap1 = new MultiValueMap<Integer, String>();
      Collection<Object> collection0 = multiValueMap1.values();
      Integer integer0 = new Integer((-2));
      multiValueMap0.put(collection0, integer0);
      ArrayList<MultiValueMap<String, Object>> arrayList0 = new ArrayList<MultiValueMap<String, Object>>();
      Factory<Collection<MultiValueMap<String, Object>>> factory1 = ConstantFactory.constantFactory((Collection<MultiValueMap<String, Object>>) arrayList0);
      MultiValueMap<Collection<Object>, MultiValueMap<String, Object>> multiValueMap2 = MultiValueMap.multiValueMap((Map<Collection<Object>, ? super Collection<MultiValueMap<String, Object>>>) multiValueMap0, factory1);
      MultiValueMap<Object, Integer> multiValueMap3 = new MultiValueMap<Object, Integer>();
      MultiValueMap<Object, String> multiValueMap4 = MultiValueMap.multiValueMap((Map<Object, ? super Collection<String>>) multiValueMap3);
      multiValueMap4.putAll((Map<?, ?>) multiValueMap2);
      assertEquals(1, linkedList0.size());
      assertFalse(multiValueMap3.isEmpty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MultiValueMap<ArrayList<Object>, Collection<Integer>> multiValueMap0 = new MultiValueMap<ArrayList<Object>, Collection<Integer>>();
      MultiValueMap<ArrayList<Object>, Integer> multiValueMap1 = MultiValueMap.multiValueMap((Map<ArrayList<Object>, ? super Collection<Integer>>) multiValueMap0);
      multiValueMap1.putAll((Map<? extends ArrayList<Object>, ?>) multiValueMap0);
      assertEquals(0, multiValueMap0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MultiValueMap<String, String> multiValueMap0 = new MultiValueMap<String, String>();
      Iterator<Map.Entry<String, String>> iterator0 = (Iterator<Map.Entry<String, String>>)multiValueMap0.iterator();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MultiValueMap<ArrayList<String>, HashMap<Object, Integer>> multiValueMap0 = new MultiValueMap<ArrayList<String>, HashMap<Object, Integer>>();
      MultiValueMap<ArrayList<String>, Integer> multiValueMap1 = MultiValueMap.multiValueMap((Map<ArrayList<String>, ? super Collection<Integer>>) multiValueMap0);
      multiValueMap1.clear();
      assertTrue(multiValueMap1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MultiValueMap<String, Collection<Object>> multiValueMap0 = new MultiValueMap<String, Collection<Object>>();
      Collection<Object> collection0 = multiValueMap0.values();
      assertNotNull(collection0);
      
      HashMap<ArrayList<Object>, Object> hashMap0 = new HashMap<ArrayList<Object>, Object>();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      Factory<LinkedList<Integer>> factory0 = ConstantFactory.constantFactory(linkedList0);
      MultiValueMap<ArrayList<Object>, Object> multiValueMap1 = new MultiValueMap<ArrayList<Object>, Object>((Map<ArrayList<Object>, ? super LinkedList<Integer>>) hashMap0, factory0);
      ArrayList<Object> arrayList0 = new ArrayList<Object>();
      boolean boolean0 = multiValueMap1.putAll(arrayList0, collection0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MultiValueMap<String, Collection<Object>> multiValueMap0 = new MultiValueMap<String, Collection<Object>>();
      Collection<Object> collection0 = multiValueMap0.values();
      multiValueMap0.put("", "");
      HashMap<ArrayList<Object>, Object> hashMap0 = new HashMap<ArrayList<Object>, Object>();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      Factory<LinkedList<Integer>> factory0 = ConstantFactory.constantFactory(linkedList0);
      MultiValueMap<ArrayList<Object>, Object> multiValueMap1 = new MultiValueMap<ArrayList<Object>, Object>((Map<ArrayList<Object>, ? super LinkedList<Integer>>) hashMap0, factory0);
      ArrayList<Object> arrayList0 = new ArrayList<Object>();
      boolean boolean0 = multiValueMap1.putAll(arrayList0, collection0);
      assertEquals(1, multiValueMap0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HashMap<String, Object> hashMap0 = new HashMap<String, Object>();
      MultiValueMap<String, AbstractMap.SimpleImmutableEntry<Collection<Object>, Integer>> multiValueMap0 = null;
      try {
        multiValueMap0 = new MultiValueMap<String, AbstractMap.SimpleImmutableEntry<Collection<Object>, Integer>>((Map<String, ? super Collection<Object>>) hashMap0, (Factory<Collection<Object>>) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The factory must not be null
         //
         verifyException("org.apache.commons.collections4.map.MultiValueMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MultiValueMap<Object, AbstractMap.SimpleImmutableEntry<Integer, Object>> multiValueMap0 = new MultiValueMap<Object, AbstractMap.SimpleImmutableEntry<Integer, Object>>();
      ArrayList<String> arrayList0 = new ArrayList<String>();
      Object object0 = multiValueMap0.put(arrayList0, arrayList0);
      Factory<ArrayList<String>> factory0 = ConstantFactory.constantFactory(arrayList0);
      MultiValueMap<Object, String> multiValueMap1 = new MultiValueMap<Object, String>((Map<Object, ? super ArrayList<String>>) multiValueMap0, factory0);
      MultiValueMap<Object, HashMap<Object, Object>> multiValueMap2 = MultiValueMap.multiValueMap((Map<Object, ? super Collection<HashMap<Object, Object>>>) multiValueMap0);
      Collection<Object> collection0 = multiValueMap2.values();
      boolean boolean0 = multiValueMap1.removeMapping(object0, collection0);
      assertEquals(1, multiValueMap0.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HashMap<Collection<Object>, Object> hashMap0 = new HashMap<Collection<Object>, Object>();
      MultiValueMap<Collection<Object>, AbstractMap.SimpleImmutableEntry<Object, Object>> multiValueMap0 = MultiValueMap.multiValueMap((Map<Collection<Object>, ? super Collection<AbstractMap.SimpleImmutableEntry<Object, Object>>>) hashMap0);
      boolean boolean0 = multiValueMap0.removeMapping((Object) null, (Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MultiValueMap<ArrayList<String>, LinkedList<Object>> multiValueMap0 = new MultiValueMap<ArrayList<String>, LinkedList<Object>>();
      ArrayList<String> arrayList0 = new ArrayList<String>();
      Object object0 = multiValueMap0.put(arrayList0, arrayList0);
      assertFalse(multiValueMap0.isEmpty());
      
      boolean boolean0 = multiValueMap0.removeMapping(object0, object0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MultiValueMap<ArrayList<String>, LinkedList<Object>> multiValueMap0 = new MultiValueMap<ArrayList<String>, LinkedList<Object>>();
      ArrayList<String> arrayList0 = new ArrayList<String>();
      BiFunction<Object, Object, Object> biFunction0 = (BiFunction<Object, Object, Object>) mock(BiFunction.class, new ViolatedAssumptionAnswer());
      doReturn(arrayList0).when(biFunction0).apply(any() , any());
      multiValueMap0.put(arrayList0, arrayList0);
      Object object0 = multiValueMap0.compute(arrayList0, biFunction0);
      boolean boolean0 = multiValueMap0.removeMapping(object0, object0);
      assertFalse(multiValueMap0.isEmpty());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MultiValueMap<ArrayList<Integer>, Object> multiValueMap0 = new MultiValueMap<ArrayList<Integer>, Object>();
      MultiValueMap<ArrayList<Integer>, LinkedList<Object>> multiValueMap1 = MultiValueMap.multiValueMap((Map<ArrayList<Integer>, ? super Collection<LinkedList<Object>>>) multiValueMap0);
      ArrayList<Integer> arrayList0 = new ArrayList<Integer>();
      multiValueMap1.putIfAbsent(arrayList0, multiValueMap0);
      boolean boolean0 = multiValueMap1.containsValue((Object) multiValueMap0);
      assertFalse(multiValueMap1.isEmpty());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MultiValueMap<LinkedList<Object>, ArrayList<Object>> multiValueMap0 = new MultiValueMap<LinkedList<Object>, ArrayList<Object>>();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      multiValueMap0.putIfAbsent(linkedList0, linkedList0);
      ArrayList<Object> arrayList0 = new ArrayList<Object>();
      boolean boolean0 = multiValueMap0.containsValue((Object) arrayList0);
      assertEquals(1, multiValueMap0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HashMap<Collection<Object>, Object> hashMap0 = new HashMap<Collection<Object>, Object>();
      LinkedList<HashMap<Collection<Object>, String>> linkedList0 = new LinkedList<HashMap<Collection<Object>, String>>();
      Factory<Collection<HashMap<Collection<Object>, String>>> factory0 = ConstantFactory.constantFactory((Collection<HashMap<Collection<Object>, String>>) linkedList0);
      MultiValueMap<Collection<Object>, HashMap<Collection<Object>, String>> multiValueMap0 = MultiValueMap.multiValueMap((Map<Collection<Object>, ? super Collection<HashMap<Collection<Object>, String>>>) hashMap0, factory0);
      MultiValueMap<Integer, String> multiValueMap1 = new MultiValueMap<Integer, String>();
      Collection<Object> collection0 = multiValueMap1.values();
      Integer integer0 = new Integer((-2));
      multiValueMap0.put(collection0, integer0);
      ArrayList<MultiValueMap<String, Object>> arrayList0 = new ArrayList<MultiValueMap<String, Object>>();
      Factory<Collection<MultiValueMap<String, Object>>> factory1 = ConstantFactory.constantFactory((Collection<MultiValueMap<String, Object>>) arrayList0);
      MultiValueMap<Collection<Object>, MultiValueMap<String, Object>> multiValueMap2 = MultiValueMap.multiValueMap((Map<Collection<Object>, ? super Collection<MultiValueMap<String, Object>>>) multiValueMap0, factory1);
      multiValueMap2.putAll((Map<? extends Collection<Object>, ?>) hashMap0);
      assertEquals(1, multiValueMap0.size());
      assertTrue(arrayList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MultiValueMap<Integer, LinkedList<Object>> multiValueMap0 = new MultiValueMap<Integer, LinkedList<Object>>();
      MultiValueMap<Integer, Object> multiValueMap1 = MultiValueMap.multiValueMap((Map<Integer, ? super Collection<Object>>) multiValueMap0);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      Factory<LinkedList<Object>> factory0 = ConstantFactory.constantFactory(linkedList0);
      Collection<Object> collection0 = multiValueMap1.values();
      assertNotNull(collection0);
      
      MultiValueMap<Integer, Integer> multiValueMap2 = new MultiValueMap<Integer, Integer>((Map<Integer, ? super LinkedList<Object>>) multiValueMap1, factory0);
      int int0 = multiValueMap2.totalSize();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Factory<Collection<AbstractMap.SimpleEntry<Object, Object>>> factory0 = ExceptionFactory.exceptionFactory();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      MultiValueMap<String, String> multiValueMap0 = new MultiValueMap<String, String>();
      boolean boolean0 = multiValueMap0.containsValue((Object) factory0, (Object) linkedList0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      MultiValueMap<String, Collection<Object>> multiValueMap0 = new MultiValueMap<String, Collection<Object>>();
      multiValueMap0.put("org.apache.commons.collections4.FunctorException", "org.apache.commons.collections4.FunctorException");
      MultiValueMap<String, Object> multiValueMap1 = MultiValueMap.multiValueMap((Map<String, ? super Collection<Object>>) multiValueMap0);
      int int0 = multiValueMap1.size((Object) "org.apache.commons.collections4.FunctorException");
      assertEquals(1, multiValueMap0.size());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HashMap<Collection<Object>, Object> hashMap0 = new HashMap<Collection<Object>, Object>();
      Factory<Collection<AbstractMap.SimpleEntry<Object, Object>>> factory0 = ExceptionFactory.exceptionFactory();
      MultiValueMap<Collection<Object>, AbstractMap.SimpleEntry<Object, Object>> multiValueMap0 = MultiValueMap.multiValueMap((Map<Collection<Object>, ? super Collection<AbstractMap.SimpleEntry<Object, Object>>>) hashMap0, factory0);
      int int0 = multiValueMap0.size((Object) hashMap0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MultiValueMap<Collection<Object>, Integer> multiValueMap0 = new MultiValueMap<Collection<Object>, Integer>();
      MultiValueMap<Object, Object> multiValueMap1 = new MultiValueMap<Object, Object>();
      Factory<Collection<String>> factory0 = ExceptionFactory.exceptionFactory();
      MultiValueMap<Object, String> multiValueMap2 = MultiValueMap.multiValueMap((Map<Object, ? super Collection<String>>) multiValueMap1, factory0);
      boolean boolean0 = multiValueMap2.putAll((Object) multiValueMap0, (Collection<String>) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      MultiValueMap<String, String> multiValueMap0 = new MultiValueMap<String, String>();
      Function<String, String> function0 = Function.identity();
      multiValueMap0.computeIfAbsent(";kCntG", function0);
      multiValueMap0.iterator((Object) ";kCntG");
      assertFalse(multiValueMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      MultiValueMap<AbstractMap.SimpleImmutableEntry<Integer, Object>, Object> multiValueMap0 = new MultiValueMap<AbstractMap.SimpleImmutableEntry<Integer, Object>, Object>();
      ArrayList<Object> arrayList0 = new ArrayList<Object>();
      Iterator<Object> iterator0 = multiValueMap0.iterator((Object) arrayList0);
      assertNotNull(iterator0);
  }
}
