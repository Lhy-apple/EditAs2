/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:53:49 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Entities;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Entities_ESTest extends Entities_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String string0 = Entities.unescape("&nbsp;");
      assertEquals("\u00A0", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity("notgreatertilde");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity("quot");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      boolean boolean0 = Entities.isBaseNamedEntity("notgreatertilde");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "H");
      Entities.escape(stringBuilder0, "       w  ", document_OutputSettings0, true, true, true);
      assertEquals("Hw ", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      // Undeclared exception!
      try { 
        Entities.escape((StringBuilder) null, " ~H)\"dGzIGXz93", document_OutputSettings0, false, true, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Entities", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder("xO<>o4f\"{@uqA");
      Entities.escape(stringBuilder0, "xO<>o4f\"{@uqA", document_OutputSettings0, true, true, true);
      assertEquals("xO<>o4f\"{@uqAxO<>o4f&quot;{@uqA", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("xO\"P&ltR9Nh*Z;H", document_OutputSettings0);
      assertEquals("xO\"P&amp;ltR9Nh*Z;H", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("fQoC^<Mg%THaF2nTmzy", document_OutputSettings0);
      assertEquals("fQoC^&lt;Mg%THaF2nTmzy", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("]]>", document_OutputSettings0);
      assertEquals("]]&gt;", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.charset("uS-ASCII");
      String string0 = Entities.escape("uS-ASCII", document_OutputSettings0);
      assertEquals("uS-ASCII", string0);
  }
}