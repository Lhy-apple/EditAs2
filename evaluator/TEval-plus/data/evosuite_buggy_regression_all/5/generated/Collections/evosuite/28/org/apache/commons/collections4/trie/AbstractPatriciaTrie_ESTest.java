/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:50:26 GMT 2023
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
import java.util.Map;
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
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      SortedMap<String, Integer> sortedMap0 = patriciaTrie0.headMap("lRmf");
      assertEquals(0, sortedMap0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>(hashMap0);
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      OrderedMapIterator<String, Object> orderedMapIterator0 = patriciaTrie1.mapIterator();
      assertFalse(orderedMapIterator0.hasPrevious());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      SortedMap<String, Object> sortedMap0 = patriciaTrie0.tailMap("T/%-GeN4o+E}YDu&]");
      assertTrue(sortedMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>(hashMap0);
      Comparator<? super String> comparator0 = patriciaTrie0.comparator();
      assertNotNull(comparator0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      SortedMap<String, Integer> sortedMap0 = patriciaTrie0.subMap("wH&Hi}Vh6g71a=Cp;", "wH&Hi}Vh6g71a=Cp;");
      assertEquals(0, sortedMap0.size());
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
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      SortedMap<String, String> sortedMap0 = patriciaTrie0.prefixMap("2 F*");
      Comparable<Object> comparable0 = (Comparable<Object>) mock(Comparable.class, new ViolatedAssumptionAnswer());
      doReturn("2 F*").when(comparable0).toString();
      AbstractPatriciaTrie.TrieEntry<Object, Comparable<Object>> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<Object, Comparable<Object>>(sortedMap0, comparable0, (-458));
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("Entry(key={} [-458], value=2 F*, parent=null, left={} [-458], right=null, predecessor={} [-458])", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.clear();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("S6)%QCd>}|J*#1U", (Integer) null);
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      Object object0 = patriciaTrie1.remove((Object) "org.apache.commons.collections4.trie.AbstractBitwiseTrie$BasicEntry");
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      SortedMap<String, String> sortedMap0 = patriciaTrie0.prefixMap("2 F*");
      Comparable<Object> comparable0 = (Comparable<Object>) mock(Comparable.class, new ViolatedAssumptionAnswer());
      doReturn("2 F*").when(comparable0).toString();
      patriciaTrie0.put("2 F*", "2 F*");
      AbstractPatriciaTrie.TrieEntry<Object, Comparable<Object>> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<Object, Comparable<Object>>(sortedMap0, comparable0, (-458));
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertFalse(sortedMap0.isEmpty());
      assertEquals("Entry(key={2 F*=2 F*} [-458], value=2 F*, parent=null, left={2 F*=2 F*} [-458], right=null, predecessor={2 F*=2 F*} [-458])", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("%/}t&", "%/}t&");
      Object object0 = patriciaTrie0.selectValue("");
      assertEquals("%/}t&", object0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      // Undeclared exception!
      try { 
        patriciaTrie0.put((String) null, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Key cannot be null
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("", (Object) null);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>();
      patriciaTrie1.put("", patriciaTrie0);
      Object object0 = patriciaTrie1.put("", patriciaTrie0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = patriciaTrie0.put("S6)%QCd>}|J*#1U", (Integer) null);
      patriciaTrie0.put(";',:TlV{C$jlYR", integer0);
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      String string0 = patriciaTrie0.getOrDefault((Object) null, "3cX^]ubN.`K|^=");
      assertEquals("3cX^]ubN.`K|^=", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("D9&{QHnxb)i", (Integer) null);
      Integer integer0 = patriciaTrie0.replace("D9&{QHnxb)i", (Integer) null);
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = patriciaTrie0.getOrDefault("wH&Hi}Vh6|71a=Cp;", (Integer) null);
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("XYHr", "XYHr");
      String string0 = patriciaTrie0.replace("@C0KuN#zGj:'RG1JpEL", "XYHr");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      String string0 = patriciaTrie0.selectKey("wH&Hi}Vh6|71a=Cp;");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("H&Hi}Vh671a=Cp;", (Integer) null);
      String string0 = patriciaTrie0.selectKey("org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeEntryMap");
      assertEquals("H&Hi}Vh671a=Cp;", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      String string0 = patriciaTrie0.selectValue("X7#BY\"od");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer((-894));
      patriciaTrie0.put("rY*CXb~D`/;:#cS[", integer0);
      patriciaTrie0.put("lRmf", (Integer) null);
      Map.Entry<String, Integer> map_Entry0 = patriciaTrie0.select("lRmf");
      assertNotNull(map_Entry0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      PatriciaTrie<String> patriciaTrie1 = new PatriciaTrie<String>(patriciaTrie0);
      PatriciaTrie<Object> patriciaTrie2 = new PatriciaTrie<Object>(patriciaTrie0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.keySet();
      Set<String> set0 = patriciaTrie0.keySet();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      Collection<String> collection0 = patriciaTrie0.values();
      Collection<String> collection1 = patriciaTrie0.values();
      assertSame(collection1, collection0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      Object object0 = patriciaTrie0.remove((Object) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("wH&Hi}Vh6|71a=Cp;", (Integer) null);
      Integer integer0 = patriciaTrie0.remove((Object) "wH&Hi}Vh6|71a=Cp;");
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      Integer integer0 = new Integer(21);
      hashMap0.put("org.apache.commons.collections4.trie.analyzer.StringKeyAnalyzer", integer0);
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>(hashMap0);
      Integer integer1 = patriciaTrie0.remove((Object) "");
      assertNull(integer1);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      patriciaTrie1.put("", "");
      Object object0 = patriciaTrie1.remove((Object) "");
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Object>("~Z_R*[PI", patriciaTrie0, (-2141));
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry1 = new AbstractPatriciaTrie.TrieEntry<String, Object>("~Z_R*[PI", "~Z_R*[PI", (-2141));
      abstractPatriciaTrie_TrieEntry0.left = abstractPatriciaTrie_TrieEntry1;
      // Undeclared exception!
      try { 
        patriciaTrie0.removeEntry(abstractPatriciaTrie_TrieEntry0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      // Undeclared exception!
      try { 
        patriciaTrie0.nextEntryImpl((AbstractPatriciaTrie.TrieEntry<String, Object>) null, (AbstractPatriciaTrie.TrieEntry<String, Object>) null, (AbstractPatriciaTrie.TrieEntry<String, Object>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>("", "GAkU6cfaI", 10);
      patriciaTrie0.addEntry(abstractPatriciaTrie_TrieEntry0, (-3978));
      // Undeclared exception!
      patriciaTrie0.higherEntry("\n");
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("_,*", "_,*");
      patriciaTrie0.put("org.apache.commons.collections4.trie.AbstractBitwiseTrie", "_,*");
      patriciaTrie0.put("&JYr", "&JYr");
      PatriciaTrie<String> patriciaTrie1 = new PatriciaTrie<String>(patriciaTrie0);
      assertTrue(patriciaTrie1.equals((Object)patriciaTrie0));
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer((-638));
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Integer>((String) null, integer0, (-638));
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry1 = new AbstractPatriciaTrie.TrieEntry<String, Integer>((String) null, integer0, 16);
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry2 = patriciaTrie0.nextEntryInSubtree(abstractPatriciaTrie_TrieEntry1, abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry2);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer(80);
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Integer>("wH&Hi}Vh6|71a=Cp;", integer0, (-719));
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.nextEntryInSubtree(abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>("", "GAkU6cfaI", 10);
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.addEntry(abstractPatriciaTrie_TrieEntry0, (-3978));
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry2 = abstractPatriciaTrie_TrieEntry0.right;
      assertNotNull(abstractPatriciaTrie_TrieEntry2);
      
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry3 = patriciaTrie0.nextEntryInSubtree(abstractPatriciaTrie_TrieEntry2, abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry3);
      assertTrue(abstractPatriciaTrie_TrieEntry2.isInternalNode());
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
      assertNotSame(abstractPatriciaTrie_TrieEntry2, abstractPatriciaTrie_TrieEntry1);
      assertFalse(abstractPatriciaTrie_TrieEntry2.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("O?", "O?");
      String string0 = patriciaTrie0.firstKey();
      assertEquals("O?", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
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
  public void test39()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("%}ft&", "%}ft&");
      String string0 = patriciaTrie0.lastKey();
      assertEquals("%}ft&", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      String string0 = patriciaTrie0.nextKey("lRmf");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      PatriciaTrie<AbstractMap.SimpleEntry<String, String>> patriciaTrie0 = new PatriciaTrie<AbstractMap.SimpleEntry<String, String>>();
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
  public void test42()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("lRmf", (Integer) null);
      String string0 = patriciaTrie0.nextKey("lRmf");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      AbstractMap.SimpleEntry<String, String> abstractMap_SimpleEntry0 = new AbstractMap.SimpleEntry<String, String>("", "");
      PatriciaTrie<AbstractMap.SimpleEntry<String, String>> patriciaTrie0 = new PatriciaTrie<AbstractMap.SimpleEntry<String, String>>();
      patriciaTrie0.put("org.apache.commons.collections4.trie.analyzer.StringKeyAnalyzer", abstractMap_SimpleEntry0);
      patriciaTrie0.put("", abstractMap_SimpleEntry0);
      String string0 = patriciaTrie0.nextKey("");
      assertNotNull(string0);
      assertEquals("org.apache.commons.collections4.trie.analyzer.StringKeyAnalyzer", string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      String string0 = patriciaTrie0.previousKey("%/}t&");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
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
  public void test46()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>();
      patriciaTrie1.put("%/}t&", patriciaTrie0);
      String string0 = patriciaTrie1.previousKey("%/}t&");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>();
      patriciaTrie1.put("%/}t&", patriciaTrie0);
      patriciaTrie1.put("", "%/}t&");
      String string0 = patriciaTrie1.previousKey("%/}t&");
      assertEquals("", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      SortedMap<String, String> sortedMap0 = patriciaTrie0.prefixMap("");
      assertTrue(sortedMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("%/}t&", "%/}t&");
      patriciaTrie0.put("", (Object) null);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("wH&Hi}Vh6|71a=Cp;", (Integer) null);
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("wH&Hi}Vh6|71a=Cp;");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry("xgfsNgQ28");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("", "");
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry("");
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("XYHr", "XYHr");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry("XYHr");
      assertFalse(abstractPatriciaTrie_TrieEntry0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      PatriciaTrie<AbstractMap.SimpleEntry<String, Object>> patriciaTrie0 = new PatriciaTrie<AbstractMap.SimpleEntry<String, Object>>();
      AbstractPatriciaTrie.TrieEntry<String, AbstractMap.SimpleEntry<String, Object>> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.lowerEntry("!Ul<uXfre0_Wbvc");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.lowerEntry((String) null);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("%}ft&", "%}ft&");
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.lowerEntry("%}ft&");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>();
      patriciaTrie1.put("", patriciaTrie0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.floorEntry("");
      assertFalse(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("wH&Hi}Vh6|71a=Cp;", "wH&Hi}Vh6|71a=Cp;");
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("wH&Hi}Vh6|71a=Cp;");
      assertFalse(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("lRmf", (Integer) null);
      // Undeclared exception!
      try { 
        patriciaTrie0.subtree("8m{cP)_0^DSG1$U)m", 0, 604);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree((String) null, 50, 50);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      patriciaTrie1.put("", "");
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.subtree("org.apache.commons.collections4.trie.aFalyzer.StringKeyAnalyzer", 0, 1);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("", (Integer) null);
      // Undeclared exception!
      try { 
        patriciaTrie0.subtree("", 0, 0);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Integer integer0 = new Integer(224);
      patriciaTrie0.put(";',:TlV{C$jlYR", integer0);
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree("1OId", 8, 8);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.put("lRmf", (Integer) null);
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree((String) null, 0, 0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("%/}t&", "%/}t&");
      patriciaTrie0.put("%-?#h$N/5", "%/}t&");
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("GR]#Gx9sP02FA_");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertFalse(abstractPatriciaTrie_TrieEntry0.isEmpty());
      assertFalse(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>((String) null, (String) null, (-4940));
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.previousEntry(abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry1);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("%tK&", "%tK&");
      patriciaTrie0.put(",.Yj0C:/U$JyBb", (Object) null);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("!Q5G2;1)!m[4");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      patriciaTrie0.put("%/}t&", "%/}t&");
      patriciaTrie0.put("", (Object) null);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("!Q5G2;1)!m[4");
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("WPC$");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>("org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeMap", "org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeMap", 0);
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.nextEntryInSubtree((AbstractPatriciaTrie.TrieEntry<String, String>) null, abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry1);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      boolean boolean0 = AbstractPatriciaTrie.isValidUplink((AbstractPatriciaTrie.TrieEntry<?, ?>) null, (AbstractPatriciaTrie.TrieEntry<?, ?>) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      AbstractMap.SimpleImmutableEntry<String, String> abstractMap_SimpleImmutableEntry0 = new AbstractMap.SimpleImmutableEntry<String, String>("ShRJ`z{", "dq]y;dw");
      AbstractPatriciaTrie.TrieEntry<AbstractMap.SimpleImmutableEntry<String, String>, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<AbstractMap.SimpleImmutableEntry<String, String>, String>(abstractMap_SimpleImmutableEntry0, "dq]y;dw", 369);
      AbstractPatriciaTrie.TrieEntry<AbstractMap.SimpleImmutableEntry<String, String>, String> abstractPatriciaTrie_TrieEntry1 = new AbstractPatriciaTrie.TrieEntry<AbstractMap.SimpleImmutableEntry<String, String>, String>(abstractMap_SimpleImmutableEntry0, "ShRJ`z{", 2);
      abstractPatriciaTrie_TrieEntry0.left = abstractPatriciaTrie_TrieEntry1;
      assertFalse(abstractPatriciaTrie_TrieEntry0.left.isInternalNode());
      
      boolean boolean0 = abstractPatriciaTrie_TrieEntry0.isExternalNode();
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("_q*", "_q*");
      String string0 = patriciaTrie0.toString();
      assertEquals("Trie[1]={\n  Entry(key=_q* [9], value=_q*, parent=ROOT, left=ROOT, right=_q* [9], predecessor=_q* [9])\n}\n", string0);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Object>((String) null, (Object) null, 0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry1 = new AbstractPatriciaTrie.TrieEntry<String, Object>((String) null, (Object) null, 0);
      abstractPatriciaTrie_TrieEntry0.parent = abstractPatriciaTrie_TrieEntry1;
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("Entry(key=null [0], value=null, parent=null [0], left=null [0], right=null, predecessor=null [0])", string0);
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Integer>("wH&Hi}Vh6|71a=Cp;", (Integer) null, 80);
      abstractPatriciaTrie_TrieEntry0.left = null;
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("Entry(key=wH&Hi}Vh6|71a=Cp; [80], value=null, parent=null, left=null, right=null, predecessor=wH&Hi}Vh6|71a=Cp; [80])", string0);
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      AbstractMap.SimpleEntry<String, String> abstractMap_SimpleEntry0 = new AbstractMap.SimpleEntry<String, String>("", "Cannot determine prefix outside of Character boundaries");
      AbstractMap.SimpleEntry<Object, Object> abstractMap_SimpleEntry1 = new AbstractMap.SimpleEntry<Object, Object>(abstractMap_SimpleEntry0);
      AbstractPatriciaTrie.TrieEntry<String, AbstractMap.SimpleEntry<Object, Object>> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, AbstractMap.SimpleEntry<Object, Object>>("Cannot determine prefix outside of Character boundaries", abstractMap_SimpleEntry1, (-1));
      AbstractPatriciaTrie.TrieEntry<String, AbstractMap.SimpleEntry<Object, Object>> abstractPatriciaTrie_TrieEntry1 = new AbstractPatriciaTrie.TrieEntry<String, AbstractMap.SimpleEntry<Object, Object>>("", abstractMap_SimpleEntry1, (-1));
      abstractPatriciaTrie_TrieEntry0.right = abstractPatriciaTrie_TrieEntry1;
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("RootEntry(key=Cannot determine prefix outside of Character boundaries [-1], value==Cannot determine prefix outside of Character boundaries, parent=null, left=ROOT, right=ROOT, predecessor=ROOT)", string0);
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
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
  public void test81()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      // Undeclared exception!
      try { 
        patriciaTrie0.subMap("B[87+E^1Gb1g", "\n");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fromKey > toKey
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeEntryMap", e);
      }
  }
}