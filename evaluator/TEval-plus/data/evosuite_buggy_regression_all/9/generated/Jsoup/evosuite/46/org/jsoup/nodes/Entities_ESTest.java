/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:08:54 GMT 2023
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
  public void test00()  throws Throwable  {
      Character character0 = Entities.getCharacterByName((String) null);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String string0 = Entities.unescape("E+Y6S2^@61$4Fhc!S");
      assertEquals("E+Y6S2^@61$4Fhc!S", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity("w/-");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity("quot");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      boolean boolean0 = Entities.isBaseNamedEntity("w/-");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      boolean boolean0 = Entities.isBaseNamedEntity("quot");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "]kh z");
      Entities.escape(stringBuilder0, "]kh z", document_OutputSettings0, true, true, true);
      assertEquals("]kh z]kh z", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder("US-ASCII");
      Entities.escape(stringBuilder0, "   ", document_OutputSettings0, false, true, false);
      assertEquals("US-ASCII ", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Entities.escape((StringBuilder) null, " ", document_OutputSettings0, true, true, true);
      assertTrue(document_OutputSettings0.prettyPrint());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, "*0p)\"X+}>2<*&2!,f\"", document_OutputSettings0, true, true, true);
      assertEquals("*0p)&quot;X+}>2<*&amp;2!,f&quot;", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("&i><U", document_OutputSettings0);
      assertEquals("&amp;i&gt;&lt;U", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("'%*?$Ge\"Cf[koq", document_OutputSettings0);
      assertEquals("'%*?$Ge\"Cf[koq", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.charset("US-ASCII");
      String string0 = Entities.escape("US-ASCII", document_OutputSettings1);
      assertEquals("US-ASCII", string0);
  }
}