/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:35:19 GMT 2023
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
      String string0 = Entities.unescape("}*oO&amp;&gt;%;*b}bZg_p q");
      assertEquals("}*oO&>%;*b}bZg_p q", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity((String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity("RightDoubleBracket");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      boolean boolean0 = Entities.isBaseNamedEntity((String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder("}*oD&>%;*b}bZg_p q");
      Entities.escape(stringBuilder0, "}*oD&>%;*b}bZg_p q", document_OutputSettings0, true, true, true);
      assertEquals("}*oD&>%;*b}bZg_p q}*oD&amp;>%;*b}bZg_p q", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Entities.escape((StringBuilder) null, "  ", document_OutputSettings0, true, true, true);
      assertTrue(document_OutputSettings0.prettyPrint());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, "        ", document_OutputSettings0, false, true, false);
      assertEquals(" ", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, "B,7;n{h|TX\"Ao&Ihu", document_OutputSettings0, true, true, true);
      assertEquals("B,7;n{h|TX&quot;Ao&amp;Ihu", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, "A]R.O`4`eR$<", document_OutputSettings0, true, true, true);
      assertEquals("A]R.O`4`eR$<", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("Q.PzRh!<l>He[TRsj", document_OutputSettings0);
      assertEquals("Q.PzRh!&lt;l&gt;He[TRsj", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("\"Vt]Y9k~", document_OutputSettings0);
      assertEquals("\"Vt]Y9k~", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.charset("US-ASCII");
      String string0 = Entities.escape("US-ASCII", document_OutputSettings1);
      assertEquals("US-ASCII", string0);
  }
}