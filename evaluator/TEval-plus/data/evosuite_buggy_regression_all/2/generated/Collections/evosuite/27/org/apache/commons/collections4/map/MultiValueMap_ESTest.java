/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:44:33 GMT 2023
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
import java.util.function.UnaryOperator;
import org.apache.commons.collections4.Factory;
import org.apache.commons.collections4.functors.ConstantFactory;
import org.apache.commons.collections4.map.MultiValueMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MultiValueMap_ESTest extends MultiValueMap_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MultiValueMap<AbstractMap.SimpleEntry<Collection<Object>, String>, Object> multiValueMap0 = new MultiValueMap<AbstractMap.SimpleEntry<Collection<Object>, String>, Object>();
      // Undeclared exception!
      try { 
        MultiValueMap.multiValueMap((Map<AbstractMap.SimpleEntry<Collection<Object>, String>, ? super LinkedList<ConstantFactory<Integer>>>) multiValueMap0, (Factory<LinkedList<ConstantFactory<Integer>>>) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The factory must not be null
         //
         verifyException("org.apache.commons.collections4.map.MultiValueMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MultiValueMap<ArrayList<Object>, Integer> multiValueMap0 = new MultiValueMap<ArrayList<Object>, Integer>();
      Iterator<Map.Entry<ArrayList<Object>, Integer>> iterator0 = (Iterator<Map.Entry<ArrayList<Object>, Integer>>)multiValueMap0.iterator();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MultiValueMap<ArrayList<String>, Collection<Object>> multiValueMap0 = new MultiValueMap<ArrayList<String>, Collection<Object>>();
      multiValueMap0.clear();
      assertTrue(multiValueMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MultiValueMap<Integer, LinkedList<String>> multiValueMap0 = new MultiValueMap<Integer, LinkedList<String>>();
      MultiValueMap<AbstractMap.SimpleEntry<Object, String>, AbstractMap.SimpleEntry<Object, Object>> multiValueMap1 = new MultiValueMap<AbstractMap.SimpleEntry<Object, String>, AbstractMap.SimpleEntry<Object, Object>>();
      boolean boolean0 = multiValueMap1.removeMapping(multiValueMap0, multiValueMap0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MultiValueMap<String, AbstractMap.SimpleEntry<Integer, Object>> multiValueMap0 = new MultiValueMap<String, AbstractMap.SimpleEntry<Integer, Object>>();
      MultiValueMap<Object, ArrayList<String>> multiValueMap1 = new MultiValueMap<Object, ArrayList<String>>();
      multiValueMap1.put(multiValueMap0, multiValueMap0);
      boolean boolean0 = multiValueMap1.containsValue((Object) "Huw$");
      assertFalse(multiValueMap1.isEmpty());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HashMap<AbstractMap.SimpleEntry<Object, Object>, Object> hashMap0 = new HashMap<AbstractMap.SimpleEntry<Object, Object>, Object>();
      MultiValueMap<AbstractMap.SimpleEntry<Object, Object>, Object> multiValueMap0 = MultiValueMap.multiValueMap((Map<AbstractMap.SimpleEntry<Object, Object>, ? super Collection<Object>>) hashMap0);
      Integer integer0 = new Integer((-567));
      AbstractMap.SimpleImmutableEntry<Integer, Object> abstractMap_SimpleImmutableEntry0 = new AbstractMap.SimpleImmutableEntry<Integer, Object>(integer0, integer0);
      AbstractMap.SimpleImmutableEntry<Integer, Object> abstractMap_SimpleImmutableEntry1 = new AbstractMap.SimpleImmutableEntry<Integer, Object>(abstractMap_SimpleImmutableEntry0);
      AbstractMap.SimpleEntry<Object, Object> abstractMap_SimpleEntry0 = new AbstractMap.SimpleEntry<Object, Object>(abstractMap_SimpleImmutableEntry1);
      hashMap0.put(abstractMap_SimpleEntry0, (Object) null);
      MultiValueMap<AbstractMap.SimpleEntry<Object, Object>, Collection<Object>> multiValueMap1 = MultiValueMap.multiValueMap((Map<AbstractMap.SimpleEntry<Object, Object>, ? super Collection<Collection<Object>>>) multiValueMap0);
      // Undeclared exception!
      try { 
        multiValueMap1.totalSize();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.collections4.map.MultiValueMap$ValuesIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MultiValueMap<Object, ArrayList<String>> multiValueMap0 = new MultiValueMap<Object, ArrayList<String>>();
      boolean boolean0 = multiValueMap0.containsValue((Object) "Huw$");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MultiValueMap<Integer, String> multiValueMap0 = new MultiValueMap<Integer, String>();
      Integer integer0 = new Integer((-29));
      UnaryOperator<Object> unaryOperator0 = UnaryOperator.identity();
      multiValueMap0.computeIfAbsent(integer0, unaryOperator0);
      boolean boolean0 = multiValueMap0.containsValue((Object) integer0);
      assertEquals(1, multiValueMap0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MultiValueMap<Object, Collection<Integer>> multiValueMap0 = new MultiValueMap<Object, Collection<Integer>>();
      Integer integer0 = new Integer(3412);
      BiFunction<Object, Object, Integer> biFunction0 = (BiFunction<Object, Object, Integer>) mock(BiFunction.class, new ViolatedAssumptionAnswer());
      doReturn(integer0).when(biFunction0).apply(any() , any());
      multiValueMap0.putIfAbsent((Object) null, (Object) null);
      multiValueMap0.computeIfPresent((Object) null, biFunction0);
      assertEquals(1, multiValueMap0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MultiValueMap<HashMap<Object, String>, HashMap<Object, String>> multiValueMap0 = new MultiValueMap<HashMap<Object, String>, HashMap<Object, String>>();
      MultiValueMap<HashMap<Object, String>, LinkedList<Object>> multiValueMap1 = MultiValueMap.multiValueMap((Map<HashMap<Object, String>, ? super Collection<LinkedList<Object>>>) multiValueMap0);
      HashMap<Object, String> hashMap0 = new HashMap<Object, String>();
      ArrayList<String> arrayList0 = new ArrayList<String>();
      multiValueMap1.put(hashMap0, arrayList0);
      multiValueMap1.putAll((Map<? extends HashMap<Object, String>, ?>) multiValueMap0);
      assertEquals(1, multiValueMap1.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MultiValueMap<String, String> multiValueMap0 = new MultiValueMap<String, String>();
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("yw9I", "L");
      multiValueMap0.putAll((Map<? extends String, ?>) hashMap0);
      assertEquals(1, multiValueMap0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MultiValueMap<Collection<Object>, Collection<String>> multiValueMap0 = new MultiValueMap<Collection<Object>, Collection<String>>();
      Collection<Object> collection0 = multiValueMap0.values();
      Collection<Object> collection1 = multiValueMap0.values();
      assertSame(collection1, collection0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MultiValueMap<ArrayList<String>, Collection<Object>> multiValueMap0 = new MultiValueMap<ArrayList<String>, Collection<Object>>();
      ConstantFactory<Object> constantFactory0 = new ConstantFactory<Object>(multiValueMap0);
      boolean boolean0 = multiValueMap0.containsValue((Object) constantFactory0, (Object) constantFactory0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HashMap<ArrayList<String>, Object> hashMap0 = new HashMap<ArrayList<String>, Object>();
      Collection<Object> collection0 = hashMap0.values();
      hashMap0.put((ArrayList<String>) null, collection0);
      MultiValueMap<ArrayList<String>, MultiValueMap<Integer, Object>> multiValueMap0 = MultiValueMap.multiValueMap((Map<ArrayList<String>, ? super Collection<MultiValueMap<Integer, Object>>>) hashMap0);
      int int0 = multiValueMap0.size((Object) null);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MultiValueMap<HashMap<Object, String>, HashMap<Object, String>> multiValueMap0 = new MultiValueMap<HashMap<Object, String>, HashMap<Object, String>>();
      MultiValueMap<HashMap<Object, String>, LinkedList<Object>> multiValueMap1 = MultiValueMap.multiValueMap((Map<HashMap<Object, String>, ? super Collection<LinkedList<Object>>>) multiValueMap0);
      MultiValueMap<AbstractMap.SimpleImmutableEntry<Object, Integer>, ArrayList<String>> multiValueMap2 = new MultiValueMap<AbstractMap.SimpleImmutableEntry<Object, Integer>, ArrayList<String>>();
      int int0 = multiValueMap2.size((Object) multiValueMap1);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ArrayList<Object> arrayList0 = new ArrayList<Object>();
      HashMap<ArrayList<Object>, Object> hashMap0 = new HashMap<ArrayList<Object>, Object>();
      Factory<Collection<Collection<Object>>> factory0 = ConstantFactory.constantFactory((Collection<Collection<Object>>) null);
      MultiValueMap<ArrayList<Object>, HashMap<Collection<Object>, Collection<Object>>> multiValueMap0 = new MultiValueMap<ArrayList<Object>, HashMap<Collection<Object>, Collection<Object>>>((Map<ArrayList<Object>, ? super Collection<Collection<Object>>>) hashMap0, factory0);
      boolean boolean0 = multiValueMap0.putAll(arrayList0, (Collection<HashMap<Collection<Object>, Collection<Object>>>) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      MultiValueMap<Collection<Object>, Collection<String>> multiValueMap0 = new MultiValueMap<Collection<Object>, Collection<String>>();
      Collection<Object> collection0 = multiValueMap0.values();
      assertNotNull(collection0);
      
      MultiValueMap<Collection<Object>, String> multiValueMap1 = new MultiValueMap<Collection<Object>, String>();
      MultiValueMap<AbstractMap.SimpleEntry<Object, Integer>, String> multiValueMap2 = new MultiValueMap<AbstractMap.SimpleEntry<Object, Integer>, String>();
      Collection<String> collection1 = multiValueMap2.createCollection((-542));
      boolean boolean0 = multiValueMap1.putAll(collection0, collection1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      MultiValueMap<Integer, Integer> multiValueMap0 = new MultiValueMap<Integer, Integer>();
      Integer integer0 = new Integer(4047);
      linkedList0.addLast(integer0);
      boolean boolean0 = multiValueMap0.putAll(integer0, (Collection<Integer>) linkedList0);
      assertEquals(1, linkedList0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MultiValueMap<AbstractMap.SimpleImmutableEntry<Object, String>, ArrayList<Collection<Object>>> multiValueMap0 = new MultiValueMap<AbstractMap.SimpleImmutableEntry<Object, String>, ArrayList<Collection<Object>>>();
      HashMap<Integer, Object> hashMap0 = new HashMap<Integer, Object>();
      MultiValueMap<Integer, ConstantFactory<Object>> multiValueMap1 = MultiValueMap.multiValueMap((Map<Integer, ? super Collection<ConstantFactory<Object>>>) hashMap0);
      MultiValueMap<Integer, String> multiValueMap2 = MultiValueMap.multiValueMap((Map<Integer, ? super Collection<String>>) multiValueMap1);
      Iterator<ArrayList<Collection<Object>>> iterator0 = multiValueMap0.iterator((Object) multiValueMap2);
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      MultiValueMap<String, String> multiValueMap0 = new MultiValueMap<String, String>();
      BiFunction<String, Object, Object> biFunction0 = (BiFunction<String, Object, Object>) mock(BiFunction.class, new ViolatedAssumptionAnswer());
      doReturn(multiValueMap0).when(biFunction0).apply(anyString() , any());
      multiValueMap0.compute("", biFunction0);
      int int0 = multiValueMap0.totalSize();
      assertFalse(multiValueMap0.isEmpty());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      MultiValueMap<AbstractMap.SimpleImmutableEntry<Object, Object>, Object> multiValueMap0 = new MultiValueMap<AbstractMap.SimpleImmutableEntry<Object, Object>, Object>();
      MultiValueMap<AbstractMap.SimpleImmutableEntry<Object, Object>, Collection<String>> multiValueMap1 = MultiValueMap.multiValueMap((Map<AbstractMap.SimpleImmutableEntry<Object, Object>, ? super Collection<Collection<String>>>) multiValueMap0);
      int int0 = multiValueMap1.totalSize();
      assertEquals(0, int0);
  }
}