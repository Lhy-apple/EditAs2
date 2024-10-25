/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:18:03 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringWriter;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attribute;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Attributes_ESTest extends Attributes_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.toString();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.dataset();
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.hashCode();
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.clone();
      boolean boolean0 = attributes1.equals(attributes0);
      assertNotSame(attributes1, attributes0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("org.jsoup.nodes.Document$OutputSettings", "org.jsoup.nodes.Document$OutputSettings");
      Attribute attribute0 = new Attribute("FiD\"l_", "");
      Attributes attributes2 = attributes1.put(attribute0);
      attributes1.putIgnoreCase("h*M;!_OD*q8#n_[", "org.jsoup.nodes.Attributes$Dataset$EntrySet");
      attributes1.addAll(attributes2);
      assertEquals(3, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      assertNotNull(attributes0);
      assertEquals(0, attributes0.size());
      
      Attributes attributes1 = attributes0.put("org.jsoup.nodes.Document$OutputSettings", "org.jsoup.nodes.Document$OutputSettings");
      assertNotNull(attributes1);
      assertEquals(1, attributes0.size());
      assertEquals(1, attributes1.size());
      assertSame(attributes0, attributes1);
      assertSame(attributes1, attributes0);
      
      Attributes attributes2 = attributes1.put("", "");
      assertNotNull(attributes2);
      assertEquals(2, attributes0.size());
      assertEquals(2, attributes1.size());
      assertEquals(2, attributes2.size());
      assertSame(attributes0, attributes2);
      assertSame(attributes0, attributes1);
      assertSame(attributes1, attributes0);
      assertSame(attributes1, attributes2);
      assertSame(attributes2, attributes1);
      assertSame(attributes2, attributes0);
      
      Attributes attributes3 = attributes1.clone();
      assertNotNull(attributes3);
      assertEquals(2, attributes0.size());
      assertEquals(2, attributes1.size());
      assertEquals(2, attributes3.size());
      assertFalse(attributes3.equals((Object)attributes1));
      assertFalse(attributes3.equals((Object)attributes0));
      assertFalse(attributes3.equals((Object)attributes2));
      assertSame(attributes0, attributes2);
      assertSame(attributes0, attributes1);
      assertNotSame(attributes0, attributes3);
      assertSame(attributes1, attributes0);
      assertSame(attributes1, attributes2);
      assertNotSame(attributes1, attributes3);
      assertNotSame(attributes3, attributes1);
      assertNotSame(attributes3, attributes0);
      assertNotSame(attributes3, attributes2);
      
      Attributes attributes4 = attributes3.put("kXR|`GOk{O", "");
      assertNotNull(attributes4);
      assertEquals(2, attributes0.size());
      assertEquals(2, attributes1.size());
      assertEquals(3, attributes3.size());
      assertEquals(3, attributes4.size());
      assertFalse(attributes0.equals((Object)attributes3));
      assertFalse(attributes1.equals((Object)attributes3));
      assertFalse(attributes3.equals((Object)attributes1));
      assertFalse(attributes3.equals((Object)attributes0));
      assertFalse(attributes3.equals((Object)attributes2));
      assertFalse(attributes4.equals((Object)attributes2));
      assertFalse(attributes4.equals((Object)attributes0));
      assertFalse(attributes4.equals((Object)attributes1));
      assertSame(attributes0, attributes2);
      assertSame(attributes0, attributes1);
      assertNotSame(attributes0, attributes3);
      assertNotSame(attributes0, attributes4);
      assertSame(attributes1, attributes0);
      assertSame(attributes1, attributes2);
      assertNotSame(attributes1, attributes4);
      assertNotSame(attributes1, attributes3);
      assertSame(attributes3, attributes4);
      assertNotSame(attributes3, attributes1);
      assertNotSame(attributes3, attributes0);
      assertNotSame(attributes3, attributes2);
      assertNotSame(attributes4, attributes2);
      assertSame(attributes4, attributes3);
      assertNotSame(attributes4, attributes0);
      assertNotSame(attributes4, attributes1);
      
      // Undeclared exception!
      try { 
        attributes1.addAll(attributes4);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      assertNotNull(attributes0);
      assertEquals(0, attributes0.size());
      
      Attributes attributes1 = attributes0.put("E#gPd`aJkpVtI{.", "E#gPd`aJkpVtI{.");
      assertNotNull(attributes1);
      assertEquals(1, attributes0.size());
      assertEquals(1, attributes1.size());
      assertSame(attributes0, attributes1);
      assertSame(attributes1, attributes0);
      
      attributes1.putIgnoreCase("E#gPd`aJkpVtI{.", "");
      assertEquals(1, attributes0.size());
      assertEquals(1, attributes1.size());
      assertSame(attributes0, attributes1);
      assertSame(attributes1, attributes0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("org.jsoup.nodes.Attributes$Dataset$DatasetIterator", true);
      attributes0.addAll(attributes1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("vwi-^{l{]s#'nk(", "vwi-^{l{]s#'nk(");
      attributes0.get("vwi-^{l{]s#'nk(");
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.get("!md|MUogc6H+((KITb");
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", "");
      attributes1.getIgnoreCase("");
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.getIgnoreCase("");
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("E#gPd`aJkpVtI{.", "E#gPd`aJkpVtI{.");
      attributes0.normalize();
      attributes1.putIgnoreCase("E#gPd`aJkpVtI{.", "");
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("org.jsoup.nodes.Document$OutputSettings", false);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("org.jsoup.nodes.Document$OutputSettings", "org.jsoup.nodes.Document$OutputSettings");
      attributes1.put("org.jsoup.nodes.Document$OutputSettings", false);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("E#gPd`ag JkpVtI{.", "E#gPd`ag JkpVtI{.");
      attributes1.put("XvZX", "XvZX");
      attributes1.remove("E#gPd`ag JkpVtI{.");
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.removeIgnoreCase("*#T`R*K;]e_{w(,EfNa");
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("zfREd=Ki", "zfREd=Ki");
      attributes1.removeIgnoreCase("zfREd=Ki");
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.hasKey("1uzw:");
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", "");
      attributes1.hasKey("");
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.hasKeyIgnoreCase("r#gPd`a/g*JkpVtI{..");
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("PLk", "PLk");
      attributes1.hasKeyIgnoreCase("PLk");
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.addAll(attributes0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("org.jsoup.nodes.Document$OutputSettings", "org.jsoup.nodes.Document$OutputSettings");
      attributes0.asList();
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("G4`#6[;xl", (String) null);
      attributes0.asList();
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("zfREd=Ki", "zfREd=Ki");
      attributes0.html();
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("z\"[BTCAS=;b&!}G4$", (String) null);
      attributes0.html();
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("CJX$ldMa2", true);
      StringWriter stringWriter0 = new StringWriter();
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      attributes1.html((Appendable) stringWriter0, document_OutputSettings1);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("FiD\"l_", "");
      attributes0.equals(attribute0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.equals(attributes0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.equals((Object) null);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = new Attributes();
      Attributes attributes2 = attributes1.put("T", "T");
      attributes2.equals(attributes0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", true);
      Attributes attributes2 = attributes1.clone();
      attributes2.equals(attributes0);
  }
}
