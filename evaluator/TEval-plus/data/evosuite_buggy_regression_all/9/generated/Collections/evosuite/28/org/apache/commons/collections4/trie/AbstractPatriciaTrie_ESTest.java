/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:20:28 GMT 2023
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
import java.util.function.BiFunction;
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
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      SortedMap<String, String> sortedMap0 = patriciaTrie0.headMap("khgY-BqN7KHC`hT");
      assertEquals(0, sortedMap0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      Set<Map.Entry<String, String>> set0 = (Set<Map.Entry<String, String>>)patriciaTrie0.entrySet();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      OrderedMapIterator<String, String> orderedMapIterator0 = patriciaTrie0.mapIterator();
      assertFalse(orderedMapIterator0.hasNext());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
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
  public void test04()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      Comparator<? super String> comparator0 = patriciaTrie0.comparator();
      assertNotNull(comparator0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      // Undeclared exception!
      try { 
        patriciaTrie0.subMap("vUMl", "");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fromKey > toKey
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie$RangeEntryMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
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
  public void test07()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      SortedMap<String, Object> sortedMap0 = patriciaTrie0.prefixMap((String) null);
      assertTrue(sortedMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      patriciaTrie0.clear();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      Collection<Integer> collection0 = patriciaTrie0.values();
      AbstractMap.SimpleImmutableEntry<Object, String> abstractMap_SimpleImmutableEntry0 = new AbstractMap.SimpleImmutableEntry<Object, String>("Entry(", "Entry(");
      AbstractMap.SimpleEntry<Object, String> abstractMap_SimpleEntry0 = new AbstractMap.SimpleEntry<Object, String>(abstractMap_SimpleImmutableEntry0);
      AbstractPatriciaTrie.TrieEntry<Object, AbstractMap.SimpleEntry<Object, String>> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<Object, AbstractMap.SimpleEntry<Object, String>>(collection0, abstractMap_SimpleEntry0, (-2));
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("Entry(key=[] [-2], value=Entry(=Entry(, parent=null, left=[] [-2], right=null, predecessor=[] [-2])", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      // Undeclared exception!
      try { 
        patriciaTrie0.put((String) null, "org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Key cannot be null
         //
         verifyException("org.apache.commons.collections4.trie.AbstractPatriciaTrie", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      String string0 = patriciaTrie0.put("", (String) null);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("@gF", "@gF");
      String string0 = patriciaTrie0.put("@gF", (String) null);
      assertEquals("@gF", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("!C'fT", "org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$KeyIterator");
      patriciaTrie0.put("@gF", "@gF");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("@gF");
      assertFalse(abstractPatriciaTrie_TrieEntry0.isEmpty());
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      
      patriciaTrie0.put("~lJl", "@gF");
      patriciaTrie0.removeEntry(abstractPatriciaTrie_TrieEntry0);
      assertFalse(abstractPatriciaTrie_TrieEntry0.isExternalNode());
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("On'DQ2V[]jb'\"", "On'DQ2V[]jb'\"");
      BiFunction<Object, String, String> biFunction0 = (BiFunction<Object, String, String>) mock(BiFunction.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(biFunction0).apply(any() , anyString());
      patriciaTrie0.merge("On'DQ2V[]jb'\"", "On'DQ2V[]jb'\"", biFunction0);
      Map.Entry<String, String> map_Entry0 = patriciaTrie0.select("On'DQ2V[]jb'\"");
      assertNull(map_Entry0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      String string0 = patriciaTrie0.previousKey("Yn^v,V=sjp'X|;");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator", "org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator");
      String string0 = patriciaTrie0.replace("GYGZ", (String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      String string0 = patriciaTrie0.selectKey((String) null);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      String string0 = patriciaTrie0.selectKey("");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      String string0 = patriciaTrie0.selectValue("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      Object object0 = patriciaTrie0.selectValue("");
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("org.apache.commons.collections4.trie.PatriciaTrie", "org.apache.commons.collections4.trie.PatriciaTrie");
      Map.Entry<String, String> map_Entry0 = patriciaTrie0.select("  ");
      assertNotNull(map_Entry0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("org.apache.commons.collections4.trie.PatriciaTrie", "org.apache.commons.collections4.trie.PatriciaTrie");
      patriciaTrie0.put("  ", "  ");
      Map.Entry<String, String> map_Entry0 = patriciaTrie0.select("  ");
      assertNotNull(map_Entry0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      String string0 = patriciaTrie0.replace((String) null, "+{aAO^-!On");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      boolean boolean0 = patriciaTrie0.containsKey("~lJl");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Yn^v,V=sjp'X|;", "Yn^v,V=sjp'X|;");
      boolean boolean0 = patriciaTrie0.containsKey("Yn^v,V=sjp'X|;");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      Set<String> set0 = patriciaTrie0.keySet();
      Set<String> set1 = patriciaTrie0.keySet();
      assertSame(set1, set0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>(hashMap0);
      Collection<Integer> collection0 = patriciaTrie0.values();
      Collection<Integer> collection1 = patriciaTrie0.values();
      assertSame(collection1, collection0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      String string0 = patriciaTrie0.remove((Object) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      String string0 = patriciaTrie0.remove((Object) "b92ddp|");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Yn^v,V=sjp'X|;", "Yn^v,V=sjp'X|;");
      String string0 = patriciaTrie0.remove((Object) "");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry((String) null);
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      
      patriciaTrie0.removeEntry(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("$hq2QDGDA+!!1 _k", "");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("WQ'm]#e1;N");
      assertFalse(abstractPatriciaTrie_TrieEntry0.isEmpty());
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("@gF", "@gF");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("@gF");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      
      patriciaTrie0.put("m", "left=");
      patriciaTrie0.put("left=", "@gF");
      patriciaTrie0.removeEntry(abstractPatriciaTrie_TrieEntry0);
      assertFalse(abstractPatriciaTrie_TrieEntry0.isExternalNode());
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("@gF", "");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("@gF");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      
      patriciaTrie0.put("]={\n", (String) null);
      patriciaTrie0.put("~lJl", (String) null);
      patriciaTrie0.removeEntry(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isEmpty());
      assertFalse(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      patriciaTrie0.put("@gF", "@gF");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry("@gF");
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
      
      patriciaTrie0.put("~lJl", "@gF");
      patriciaTrie0.removeEntry(abstractPatriciaTrie_TrieEntry0);
      assertFalse(abstractPatriciaTrie_TrieEntry0.isExternalNode());
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      // Undeclared exception!
      try { 
        patriciaTrie0.nextEntryImpl((AbstractPatriciaTrie.TrieEntry<String, Integer>) null, (AbstractPatriciaTrie.TrieEntry<String, Integer>) null, (AbstractPatriciaTrie.TrieEntry<String, Integer>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet", "org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("+Sr*DOk'y5");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = new AbstractPatriciaTrie.TrieEntry<String, String>("]={\n", "org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet", (-3835));
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry2 = patriciaTrie0.nextEntryImpl(abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry1, abstractPatriciaTrie_TrieEntry0);
      assertNotNull(abstractPatriciaTrie_TrieEntry2);
      assertFalse(abstractPatriciaTrie_TrieEntry1.isInternalNode());
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
      assertTrue(abstractPatriciaTrie_TrieEntry1.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>("'Vo}NXLGg{0~%2", "M3", (-3));
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.addEntry(abstractPatriciaTrie_TrieEntry0, 32);
      assertFalse(abstractPatriciaTrie_TrieEntry1.isInternalNode());
      
      patriciaTrie0.decrementSize();
      PatriciaTrie<String> patriciaTrie1 = new PatriciaTrie<String>(patriciaTrie0);
      assertFalse(patriciaTrie1.equals((Object)patriciaTrie0));
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("The offsets and lengths must be at Character boundaries", "`=2NvYIY");
      patriciaTrie0.put("]={\n", "]={\n");
      patriciaTrie0.put("org.apache.commons.collections4.trie.AbstractBitwiseTrie", "=");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("=");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
      assertFalse(abstractPatriciaTrie_TrieEntry0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("@q9.pUM@", "@q9.pUM@");
      patriciaTrie0.put("wK7C", "b92ddp|");
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.lowerEntry("b92ddp|");
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("XaSf7\"", "XaSf7\"");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.getEntry("XaSf7\"");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.nextEntryImpl(abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry1);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("j*%QWh1c>%#", "j*%QWh1c>%#");
      String string0 = patriciaTrie0.firstKey();
      assertEquals("j*%QWh1c>%#", string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
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
  public void test44()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      String string0 = patriciaTrie0.nextKey("ES$zqoqofd+M ");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
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
  public void test46()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("ES$zqoqofd+M ", "");
      String string0 = patriciaTrie0.nextKey("ES$zqoqofd+M ");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("wK7C", "qb#92ddp|");
      patriciaTrie0.put("ES$zqoqofd+M ", "");
      String string0 = patriciaTrie0.nextKey("ES$zqoqofd+M ");
      assertEquals("wK7C", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
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
  public void test49()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("z|SH", "=");
      String string0 = patriciaTrie0.previousKey("z|SH");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("@gF", "");
      patriciaTrie0.put("", (String) null);
      String string0 = patriciaTrie0.previousKey("@gF");
      assertNotNull(string0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      SortedMap<String, Object> sortedMap0 = patriciaTrie0.prefixMap("Z4 iXCh");
      assertTrue(sortedMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry((String) null);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", (String) null);
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry((String) null);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      patriciaTrie0.put("wK7C", "");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry((String) null);
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Yn^v,V=sjp'X|;", "Yn^v,V=sjp'X|;");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.higherEntry("Yn^v,V=sjp'X|;");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry("]={\n");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry((String) null);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Yn^v,V=sjp'X|;", "Yn^v,V=sjp'X|;");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.ceilingEntry("Yn^v,V=sjp'X|;");
      assertFalse(abstractPatriciaTrie_TrieEntry0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.lowerEntry("b92ddp|");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.lowerEntry("");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("The offsets andKlengths must be at Character boundaries", "The offsets andKlengths must be at Character boundaries");
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.lowerEntry("The offsets andKlengths must be at Character boundaries");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry((String) null);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", (String) null);
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.floorEntry((String) null);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isExternalNode());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree("V}0,i#QZ.UE5J.", 64, 64);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("j*%QWh1c>%#", "j*%QWh1c>%#");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree("3cwVp@G", 0, 0);
      assertFalse(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("Yn^v,V=sjp'X|;", "Yn^v,V=sjp'X|;");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree("V}0,i#QZ.UE5J.", 64, 64);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", (String) null);
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree((String) null, (-592), 368);
      assertFalse(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree("EpbR6xj?mC", 2553, 2553);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("I7$vVzvhZmg=f~sI/t", "I7$vVzvhZmg=f~sI/t");
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.subtree((String) null, 0, 16);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("@gF", "");
      patriciaTrie0.put("]={\n", (String) null);
      String string0 = patriciaTrie0.lastKey();
      assertNotNull(string0);
      assertEquals("]={\n", string0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, Object>("", "", (-5177));
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.previousEntry(abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry1);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("jZ", "jZ");
      patriciaTrie0.put("wK7C", "b92ddp|");
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.lowerEntry("b92ddp|");
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      patriciaTrie0.put("wK7C", "b92ddp|");
      PatriciaTrie<Object> patriciaTrie1 = new PatriciaTrie<Object>(patriciaTrie0);
      AbstractPatriciaTrie.TrieEntry<String, Object> abstractPatriciaTrie_TrieEntry0 = patriciaTrie1.lowerEntry("b92ddp|");
      assertNotNull(abstractPatriciaTrie_TrieEntry0);
      assertTrue(abstractPatriciaTrie_TrieEntry0.isInternalNode());
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<String, String>("hzOO0.#)>zV(5.BR/j>", "org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator", 304);
      AbstractPatriciaTrie.TrieEntry<String, String> abstractPatriciaTrie_TrieEntry1 = patriciaTrie0.nextEntryInSubtree(abstractPatriciaTrie_TrieEntry0, abstractPatriciaTrie_TrieEntry0);
      assertNull(abstractPatriciaTrie_TrieEntry1);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      AbstractPatriciaTrie.TrieEntry<String, Integer> abstractPatriciaTrie_TrieEntry0 = patriciaTrie0.nextEntryInSubtree((AbstractPatriciaTrie.TrieEntry<String, Integer>) null, (AbstractPatriciaTrie.TrieEntry<String, Integer>) null);
      assertNull(abstractPatriciaTrie_TrieEntry0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      boolean boolean0 = AbstractPatriciaTrie.isValidUplink((AbstractPatriciaTrie.TrieEntry<?, ?>) null, (AbstractPatriciaTrie.TrieEntry<?, ?>) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Integer integer0 = new Integer((-3947));
      AbstractPatriciaTrie.TrieEntry<Integer, Object> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<Integer, Object>(integer0, integer0, (-591));
      assertFalse(abstractPatriciaTrie_TrieEntry0.isInternalNode());
      
      abstractPatriciaTrie_TrieEntry0.left = null;
      boolean boolean0 = abstractPatriciaTrie_TrieEntry0.isExternalNode();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("", "");
      String string0 = patriciaTrie0.toString();
      assertEquals("Trie[1]={\n  RootEntry(key= [-1], value=, parent=null, left=ROOT, right=null, predecessor=ROOT)\n}\n", string0);
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      PatriciaTrie<String> patriciaTrie0 = new PatriciaTrie<String>();
      patriciaTrie0.put("org.apache.commons.collections4.trie.PatriciaTrie", "org.apache.commons.collections4.trie.PatriciaTrie");
      patriciaTrie0.put("org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator", "org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator");
      String string0 = patriciaTrie0.toString();
      assertEquals("Trie[2]={\n  Entry(key=org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator [603], value=org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator, parent=org.apache.commons.collections4.trie.PatriciaTrie [9], left=org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator [603], right=org.apache.commons.collections4.trie.PatriciaTrie [9], predecessor=org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator [603])\n  Entry(key=org.apache.commons.collections4.trie.PatriciaTrie [9], value=org.apache.commons.collections4.trie.PatriciaTrie, parent=ROOT, left=ROOT, right=org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator [603], predecessor=org.apache.commons.collections4.trie.AbstractPatriciaTrie$KeySet$eyIterator [603])\n}\n", string0);
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Integer integer0 = new Integer(1996);
      AbstractPatriciaTrie.TrieEntry<Integer, Integer> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<Integer, Integer>(integer0, integer0, 1996);
      abstractPatriciaTrie_TrieEntry0.left = null;
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("Entry(key=1996 [1996], value=1996, parent=null, left=null, right=null, predecessor=1996 [1996])", string0);
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      AbstractPatriciaTrie.TrieEntry<Object, String> abstractPatriciaTrie_TrieEntry0 = new AbstractPatriciaTrie.TrieEntry<Object, String>("o=tF", "=X.P^j [?J=zuEb'K", (-4637));
      abstractPatriciaTrie_TrieEntry0.predecessor = null;
      String string0 = abstractPatriciaTrie_TrieEntry0.toString();
      assertEquals("Entry(key=o=tF [-4637], value==X.P^j [?J=zuEb'K, parent=null, left=o=tF [-4637], right=null, )", string0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      PatriciaTrie<Object> patriciaTrie0 = new PatriciaTrie<Object>();
      SortedMap<String, Object> sortedMap0 = patriciaTrie0.tailMap("");
      assertTrue(sortedMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      PatriciaTrie<Integer> patriciaTrie0 = new PatriciaTrie<Integer>();
      SortedMap<String, Integer> sortedMap0 = patriciaTrie0.subMap("", "");
      assertTrue(sortedMap0.isEmpty());
  }
}