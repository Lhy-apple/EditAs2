/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:45:17 GMT 2023
 */

package org.apache.commons.collections4.trie;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.AbstractMap;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.SortedMap;
import org.apache.commons.collections4.OrderedMapIterator;
import org.apache.commons.collections4.trie.AbstractPatriciaTrie;
import org.apache.commons.collections4.trie.PatriciaTrie;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AbstractPatriciaTrie_ESTest extends AbstractPatriciaTrie_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      SortedMap<String, Object> sortedMap0 = patriciaTrie0.headMap("Jh`;vX");
      assertTrue(sortedMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      OrderedMapIterator<String, String> orderedMapIterator0 = patriciaTrie0.mapIterator();
      assertFalse(orderedMapIterator0.hasNext());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      // Undeclared exception!
      try { 
        patriciaTrie0.tailMap((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have a from or to!
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeEntryMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      Comparator<? super String> comparator0 = patriciaTrie0.comparator();
      assertNotNull(comparator0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      SortedMap<String, Object> sortedMap0 = patriciaTrie0.subMap("Cannot determine prefix outside of Character boundaries", (String) null);
      assertTrue(sortedMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      // Undeclared exception!
      try { 
        patriciaTrie0.lastKey();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      SortedMap<String, Object> sortedMap0 = patriciaTrie1.prefixMap("");
      assertTrue(sortedMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.clear();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      Set<String> set0 = patriciaTrie0.keySet();
      Integer integer0 = new Integer((-1045));
      AbstractMap.SimpleEntry<Object, Integer> abstractMap_SimpleEntry0 = new AbstractMap.SimpleEntry<Object, Integer>(set0, integer0);
      AbstractPatriciaTrie.TrieEntry<AbstractMap.SimpleEntry<Object, Integer>, Object> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<AbstractMap.SimpleEntry<Object, Integer>, Object>(abstractMap_SimpleEntry0, "]={\n", (-1045));
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("Entry(key=[]=-1045 [-1045], value=]={\n, parent=null, left=[]=-1045 [-1045], right=null, predecessor=[]=-1045 [-1045])", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      // Undeclared exception!
      try { 
        patriciaTrie0.put((String) null, "wWDn_&\"3(}l?g&Es");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Key cannot be null
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("Cannot determine prefix outside of Character boundaries", (Object) null);
      Object object0 = patriciaTrie0.put("Cannot determine prefix outside of Character boundaries", "Cannot determine prefix outside of Character boundaries");
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PatriciaTrie<Comparable<Object>> patriciaTrie0 = new PatriciaTrie<Comparable<Object>>();
      Comparable<Object> comparable0 = (Comparable<Object>) mock(Comparable.class, new ViolatedAssumptionAnswer());
      doReturn((String) null, (String) null, (String) null, (String) null, (String) null).when(comparable0).toString();
      patriciaTrie0.put("left=", comparable0);
      AbstractPatriciaTrie.TrieEntry<String, Comparable<Object>> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.lowerEntry("value=");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
      
      AbstractPatriciaTrie.TrieEntry<String, Comparable<Object>> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.addEntry(abstractPatriciaTrie_TrieEntry0, 1);
      assertFalse(abstractPatriciaTrie_TrieEntry1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      String string0 = patriciaTrie0.replace("The offsets and lengths must be at Character boundaries", "must vave a from ortx!");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      boolean boolean0 = patriciaTrie0.remove((Object) null, (Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("fdztj'W+)2!;t 8", "fdztj'W+)2!;t 8");
      String string0 = patriciaTrie0.replace("The offsets and lengths must be at Character boundaries", "must vave a from ortx!");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>(hashMap0);
      String string0 = patriciaTrie0.selectKey("v3(s4Jg@j$`FC4;4j{$");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer(3674);
      patriciaTrie0.put("+rr5-QJ0W:(+>l:h%", integer0);
      Integer integer1 = patriciaTrie0.selectValue(">lhK");
      assertEquals(3674, (int)integer1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PatriciaTrie<AbstractMap.SimpleEntry<Integer, String>> patriciaTrie0 = new PatriciaTrie<AbstractMap.SimpleEntry<Integer, String>>();
      AbstractMap.SimpleEntry<Integer, String> abstractMap_SimpleEntry0 = patriciaTrie0.selectValue("org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeMap");
      assertNull(abstractMap_SimpleEntry0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("fdztj'W+)2!;t 8", "fdztj'W+)2!;t 8");
      patriciaTrie0.put("must vave a from ortx!", "must vave a from ortx!");
      String string0 = patriciaTrie0.selectKey((String) null);
      assertEquals("fdztj'W+)2!;t 8", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = patriciaTrie0.put("", (Integer) null);
      Integer integer1 = patriciaTrie0.replace("", integer0);
      assertNull(integer1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.keySet();
      Set<String> set0 = patriciaTrie0.keySet();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PatriciaTrie<AbstractMap.SimpleEntry<Integer, String>> patriciaTrie0 = new PatriciaTrie<AbstractMap.SimpleEntry<Integer, String>>();
      Collection<AbstractMap.SimpleEntry<Integer, String>> collection0 = patriciaTrie0.values();
      Collection<AbstractMap.SimpleEntry<Integer, String>> collection1 = patriciaTrie0.values();
      assertSame(collection1, collection0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      String string0 = patriciaTrie0.remove((Object) "Jh`;vX");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      String string0 = patriciaTrie0.remove((Object) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Jh`;vX", "Jh`;vX");
      String string0 = patriciaTrie0.remove((Object) "Jh`;vX");
      assertEquals("Jh`;vX", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>("[&}=!w29?[E-e,", "[&}=!w29?[E-e,", 1024);
      patriciaTrie0.addEntry(abstractPatriciaTrie_TrieEntry0, 1024);
      String string0 = patriciaTrie0.remove((Object) "");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      String string0 = patriciaTrie0.remove((Object) "");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Jh`;vX", "Jh`;vX");
      patriciaTrie0.put("wWDn_&\"3(}l?g&Es", "wWDn_&\"3(}l?g&Es");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("Jh`;vX");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      
      patriciaTrie0.removeEntry(abstractPatriciaTrie_TrieEntry0);
      patriciaTrie0.previousEntry(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isEmpty());
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("fdztj'W+)2!;t 8", "fdztj'W+)2!;t 8");
      patriciaTrie0.put("~@.p}6Sr", "~@.p}6Sr");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("A+-8^h\"|j");
      assertFalse(abstractPatriciaTrie_TrieEntry0.isEmpty());
      assertFalse(abstractPatriciaTrie_TrieEntry0.isExternalNode());
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Jh`;vX", "Jh`;vX");
      patriciaTrie0.put("wWDn_&\"3(}l?g&Es", "XA? Ns+.MZ<");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("Jh`;vX");
      assertFalse(abstractPatriciaTrie_TrieEntry0.isEmpty());
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      
      patriciaTrie0.put("Q", "Jh`;vX");
      String string0 = patriciaTrie0.removeEntry(abstractPatriciaTrie_TrieEntry0);
      assertEquals("Jh`;vX", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Jh`;vX", "Jh`;vX");
      patriciaTrie0.put("wWDn_&\"3(}l?g&Es", "XA? Ns+.MZ<");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("Jh`;vX");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertFalse(abstractPatriciaTrie_TrieEntry0.isEmpty());
      
      patriciaTrie0.put("", "Jh`;vX");
      patriciaTrie0.removeEntry(abstractPatriciaTrie_TrieEntry0);
      assertFalse(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>((String) null, (String) null, 807);
      // Undeclared exception!
      try { 
        patriciaTrie0.nextEntryImpl((AbstractPatriciaTrie.TrieEntry<String, String>) null, (AbstractPatriciaTrie.TrieEntry<String, String>) null, abstractPatriciaTrie_TrieEntry0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      PatriciaTrie<Comparable<Object>> patriciaTrie0 = new PatriciaTrie<Comparable<Object>>();
      Comparable<Object> comparable0 = (Comparable<Object>) mock(Comparable.class, new ViolatedAssumptionAnswer());
      doReturn((String) null, (String) null, (String) null).when(comparable0).toString();
      patriciaTrie0.put("left=", comparable0);
      AbstractPatriciaTrie.TrieEntry<String, Comparable<Object>> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Comparable<Object>>("value=", (Comparable<Object>) null, 1);
      patriciaTrie0.addEntry(abstractPatriciaTrie_TrieEntry0, 1);
      AbstractPatriciaTrie.TrieEntry<String, Comparable<Object>> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.nextEntryImpl(abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0);
      assertNotNull(abstractPatriciaTrie_TrieEntry1);
      assertFalse(abstractPatriciaTrie_TrieEntry1.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer(3732);
      Integer integer1 = patriciaTrie0.put("'Ry&", integer0);
      patriciaTrie0.put("t1&7HV@eR++.?>zVGMK", integer1);
      patriciaTrie0.put("J`B(qBfC", integer0);
      patriciaTrie0.put("L^$0FJ_VH3M`s6]T==t", integer0);
      PatriciaTrie<Integer> patriciaTrie1 = new PatriciaTrie<Integer>(patriciaTrie0);
      assertTrue(patriciaTrie1.equals((Object)patriciaTrie0));
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Object>((String) null, (Object) null, 16);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.nextEntryImpl(abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry1);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>(hashMap0);
      patriciaTrie0.put("Entry(key==0 [0], value==0, parent=null, left==0 [0], right=null, predecessor==0 [0])", "Entry(key==0 [0], value==0, parent=null, left==0 [0], right=null, predecessor==0 [0])");
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree("Entry(key==0 [0], value==0, parent=null, left==0 [0], right=null, predecessor==0 [0])", 0, 0);
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      
      patriciaTrie0.nextEntryImpl(abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0);
      assertFalse(abstractPatriciaTrie_TrieEntry0.isInternalNode());
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer(3732);
      patriciaTrie0.put("'Ry&", integer0);
      String string0 = patriciaTrie0.firstKey();
      assertEquals("'Ry&", string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      // Undeclared exception!
      try { 
        patriciaTrie0.firstKey();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      String string0 = patriciaTrie1.nextKey("");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      // Undeclared exception!
      try { 
        patriciaTrie0.nextKey((String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Integer integer0 = new Integer(1253);
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("", integer0);
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      String string0 = patriciaTrie1.nextKey("");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("", (Object) null);
      patriciaTrie0.put("Cannot determine prefix outside of Character boundaries", (Object) null);
      String string0 = patriciaTrie0.nextKey("");
      assertEquals("Cannot determine prefix outside of Character boundaries", string0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      String string0 = patriciaTrie0.previousKey("L^$0FJ_VH3M`s6]T==t");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      // Undeclared exception!
      try { 
        patriciaTrie0.previousKey((String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer(3732);
      patriciaTrie0.put("L^$0FJ_VH3M`s6]T==t", integer0);
      String string0 = patriciaTrie0.previousKey("L^$0FJ_VH3M`s6]T==t");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer(3732);
      Integer integer1 = patriciaTrie0.put("'Ry&", integer0);
      patriciaTrie0.put("L^$0FJ_VH3M`s6]T==t", integer1);
      String string0 = patriciaTrie0.previousKey("L^$0FJ_VH3M`s6]T==t");
      assertEquals("'Ry&", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      SortedMap<String, Integer> sortedMap0 = patriciaTrie0.prefixMap("{9;u");
      assertEquals(0, sortedMap0.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      patriciaTrie0.put("Jh`;vX", "Jh`;vX");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("AYDH2Au", "AYDH2Au");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("AYDH2Au");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      PatriciaTrie<AbstractMap.SimpleEntry<Integer, String>> patriciaTrie0 = new PatriciaTrie<AbstractMap.SimpleEntry<Integer, String>>();
      AbstractPatriciaTrie.TrieEntry<String, AbstractMap.SimpleEntry<Integer, String>> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry("org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeMap");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>();
      patriciaTrie1.put("", patriciaTrie0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.ceilingEntry("");
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      PatriciaTrie<AbstractMap.SimpleEntry<Integer, String>> patriciaTrie0 = new PatriciaTrie<AbstractMap.SimpleEntry<Integer, String>>();
      patriciaTrie0.put("{n#", (AbstractMap.SimpleEntry<Integer, String>) null);
      AbstractPatriciaTrie.TrieEntry<String, AbstractMap.SimpleEntry<Integer, String>> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry("{n#");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      PatriciaTrie<AbstractMap.SimpleImmutableEntry<Object, Integer>> patriciaTrie0 = new PatriciaTrie<AbstractMap.SimpleImmutableEntry<Object, Integer>>();
      AbstractPatriciaTrie.TrieEntry<String, AbstractMap.SimpleImmutableEntry<Object, Integer>> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.lowerEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer((-1607));
      patriciaTrie0.put("{9;u", integer0);
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.lowerEntry("{9;u");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "21nky{f56|5kI+BT8");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("");
      assertFalse(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("oiOACE,z31jRn");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.subtree((String) null, 0, 16);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer(16);
      patriciaTrie0.put("m[", integer0);
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      // Undeclared exception!
      try { 
        patriciaTrie1.subtree("Entry(key=144 [5390], value=m[, parent=null, left=144 [5390], right=null, predecessor=144 [5390])", 16, 5390);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("fdztj'W+)2!;t 8", "fdztj'W+)2!;t 8");
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      // Undeclared exception!
      try { 
        patriciaTrie1.subtree((String) null, 0, 16);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.collections4.trie.analyzer.StringKeyAnalyzer", e);
      }
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("", (Object) null);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree((String) null, 433, 433);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>();
      patriciaTrie1.put("", patriciaTrie0);
      // Undeclared exception!
      try { 
        patriciaTrie1.subtree("", (-1), 1);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>();
      patriciaTrie1.put("}*B-q:-#Y>C/rD{^D4&", patriciaTrie0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.subtree("?cM\"NeH|x/* _L", 0, 128);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("pz?a6?71_*yV", "pz?a6?71_*yV");
      Set<String> set0 = patriciaTrie0.keySet();
      patriciaTrie0.put("J]$nxKt1;)b\n%_/$af", set0);
      assertEquals(2, set0.size());
      
      String string0 = patriciaTrie0.lastKey();
      assertEquals("pz?a6?71_*yV", string0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("wWDn_&\"3(}l?g&Es", "wWDn_&\"3(}l?g&Es");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("Jh`;vX");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("wWDn_&\"3(}l?g&Es", "oiOACE,z31jRn");
      patriciaTrie0.put("", "");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("oiOACE,z31jRn");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Jh`;vX", "Jh`;vX");
      patriciaTrie0.put("wWDn_&\"3(}l?g&Es", (String) null);
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("org.apache.commons.collections4.trie.analyzer.StringKeyAnalyzer");
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>("[&}=!w29?[E-e,", "%PbyqW&.\"ATkN,|8<8", 1024);
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.nextEntryInSubtree(abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry1);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>((String) null, (String) null, (-3893));
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.nextEntryInSubtree((AbstractPatriciaTrie.TrieEntry<String, String>) null, abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry1);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      boolean boolean0 = AbstractPatriciaTrie.isValidUplink((AbstractPatriciaTrie.TrieEntry<?, ?>) null, (AbstractPatriciaTrie.TrieEntry<?, ?>) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Integer integer0 = new Integer(0);
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Integer>((String) null, integer0, (-1907));
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry1 = new AbstractPatriciaTrie.TrieEntry<String, Integer>("gwNm{o:{vF*9W?", integer0, (-1380));
      abstractPatriciaTrie_TrieEntry0.left = abstractPatriciaTrie_TrieEntry1;
      assertFalse(abstractPatriciaTrie_TrieEntry0.left.isInternalNode());
      
      boolean boolean0 = abstractPatriciaTrie_TrieEntry0.isExternalNode();
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Integer integer0 = new Integer((-1));
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Integer>("wsFAoS*NC", integer0, (-1));
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("RootEntry(key=wsFAoS*NC [-1], value=-1, parent=null, left=ROOT, right=null, predecessor=ROOT)", string0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("fdztj'W+)2!;t 8", "fdztj'W+)2!;t 8");
      patriciaTrie0.put("must vave a from ortx!", "must vave a from ortx!");
      AbstractPatriciaTrie.TrieEntry<Object, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<Object, String>(patriciaTrie0, (String) null, 2253);
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Integer integer0 = new Integer(0);
      AbstractMap.SimpleImmutableEntry<Object, Integer> abstractMap_SimpleImmutableEntry0 = new AbstractMap.SimpleImmutableEntry<Object, Integer>("", integer0);
      AbstractMap.SimpleImmutableEntry<Object, Integer> abstractMap_SimpleImmutableEntry1 = new AbstractMap.SimpleImmutableEntry<Object, Integer>(abstractMap_SimpleImmutableEntry0);
      AbstractMap.SimpleEntry<Object, Integer> abstractMap_SimpleEntry0 = new AbstractMap.SimpleEntry<Object, Integer>(abstractMap_SimpleImmutableEntry1);
      AbstractPatriciaTrie.TrieEntry<AbstractMap.SimpleEntry<Object, Integer>, Object> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<AbstractMap.SimpleEntry<Object, Integer>, Object>(abstractMap_SimpleEntry0, abstractMap_SimpleEntry0, (-1045));
      abstractPatriciaTrie_TrieEntry0.left = null;
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("Entry(key==0 [-1045], value==0, parent=null, left=null, right=null, predecessor==0 [-1045])", string0);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>("org.apache.commons.collections4.trie.AbstractBitwiseTrie$BasicEntry", "}\n", 2300);
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.addEntry(abstractPatriciaTrie_TrieEntry0, 128);
      assertTrue(abstractPatriciaTrie_TrieEntry1.isExternalNode());
      
      AbstractPatriciaTrie.TrieEntry<Object, Comparable<String>> abstractPatriciaTrie_TrieEntry2 = new AbstractPatriciaTrie.TrieEntry<Object, Comparable<String>>(abstractPatriciaTrie_TrieEntry0, "org.apache.commons.collections4.trie.AbstractBitwiseTrie$BasicEntry", 966);
      abstractPatriciaTrie_TrieEntry2.toString();
      assertFalse(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Integer integer0 = new Integer(1181);
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Integer>("Entry(key=3=AHzjBtH/e1r?Q)EB [1181], value=3=AHzjBtH/e1r?Q)EB, parent=null, left=3=AHzjBtH/e1r?Q)EB [1181], right=null, predecessor=3=AHzjBtH/e1r?Q)EB [1181])", integer0, 1181);
      abstractPatriciaTrie_TrieEntry0.predecessor = null;
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("Entry(key=Entry(key=3=AHzjBtH/e1r?Q)EB [1181], value=3=AHzjBtH/e1r?Q)EB, parent=null, left=3=AHzjBtH/e1r?Q)EB [1181], right=null, predecessor=3=AHzjBtH/e1r?Q)EB [1181]) [1181], value=1181, parent=null, left=Entry(key=3=AHzjBtH/e1r?Q)EB [1181], value=3=AHzjBtH/e1r?Q)EB, parent=null, left=3=AHzjBtH/e1r?Q)EB [1181], right=null, predecessor=3=AHzjBtH/e1r?Q)EB [1181]) [1181], right=null, )", string0);
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      SortedMap<String, String> sortedMap0 = patriciaTrie0.subMap("org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeEntrySet", "org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeEntrySet");
      assertEquals(0, sortedMap0.size());
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      // Undeclared exception!
      try { 
        patriciaTrie0.subMap("org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet", "/z");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fromKey > toKey
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeEntryMap", e);
      }
  }
}