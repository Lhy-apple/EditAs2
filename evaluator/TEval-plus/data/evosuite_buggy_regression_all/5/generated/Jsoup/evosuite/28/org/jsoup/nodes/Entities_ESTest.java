/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:12:41 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Entities;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Entities_ESTest extends Entities_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Character character0 = Entities.getCharacterByName((String) null);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      String string0 = Entities.unescape("pqlb]Q&vDm;(2CH)");
      assertEquals("pqlb]Q&vDm;(2CH)", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape(" />", document_OutputSettings0);
      assertEquals(" /&gt;", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity((String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity("NestedLessLess");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      String string0 = Entities.unescape("UTF-8");
      assertEquals("UTF-8", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      String string0 = Entities.unescape(" /&gt;", true);
      assertEquals(" />", string0);
  }
}