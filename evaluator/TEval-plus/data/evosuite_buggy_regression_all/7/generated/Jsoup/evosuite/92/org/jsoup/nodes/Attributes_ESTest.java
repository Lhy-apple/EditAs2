/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:29:28 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringWriter;
import java.util.List;
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
      Attributes attributes1 = attributes0.put("org.jsoup.nodes.Attributes$1", true);
      attributes0.addAll(attributes1);
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
      Attributes attributes2 = attributes1.put(")\"'YcoyNS4}lc4J", false);
      boolean boolean0 = attributes2.equals(attributes0);
      assertTrue(attributes1.equals((Object)attributes0));
      assertNotSame(attributes2, attributes0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase(" w{u7u([Es", " w{u7u([Es");
      attributes0.putIgnoreCase("=\"", " w{u7u([Es");
      attributes0.putIgnoreCase("", " w{u7u([Es");
      attributes0.putIgnoreCase("Te}mU Ns709cW{+0},", (String) null);
      attributes0.putIgnoreCase("bWJ+58-%EB_gE'WT", "bIOA/9hm");
      assertEquals(5, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", (String) null);
      Attributes attributes2 = attributes1.put(" {Y@v>KUA?", true);
      Attribute attribute0 = Attribute.createFromEncoded(" {Y@v>KUA?", " {Y@v>KUA?");
      Attributes attributes3 = attributes2.put(attribute0);
      Attributes attributes4 = attributes3.clone();
      // Undeclared exception!
      try { 
        attributes0.addAll(attributes4);
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
      attributes0.putIgnoreCase(")\"'ycoyns4}lc4j", ")\"'ycoyns4}lc4j");
      assertEquals(1, attributes0.size());
      
      attributes0.removeIgnoreCase(")\"'ycoyns4}lc4j");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase("9BY5 cnB)$]Nov ", "9BY5 cnB)$]Nov ");
      attributes0.get("9BY5 cnB)$]Nov ");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.get("(Y$p");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase("\"/", "\"/");
      attributes0.getIgnoreCase("\"/");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.getIgnoreCase("org.jsoup.nodes.Attributes");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase("=\"", " w{u7u([Es");
      attributes0.putIgnoreCase("=\"", "");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("awdu%ku([es", true);
      attributes0.putIgnoreCase("Awdu%Ku([Es", "Awdu%Ku([Es");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("\"/", false);
      assertEquals(0, attributes1.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put(")\"'YcoyNS4}lc4J", ")\"'YcoyNS4}lc4J");
      attributes1.putIgnoreCase("Awdu%Kh([En", "Awdu%Kh([En");
      attributes1.put(")\"'YcoyNS4}lc4J", false);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.removeIgnoreCase("'");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKey("SOY]hnKrx0lQ6t+rOje");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put(")\"'YcoyNS4}lc4J", ")\"'YcoyNS4}lc4J");
      boolean boolean0 = attributes1.hasKey(")\"'YcoyNS4}lc4J");
      assertEquals(1, attributes0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKeyIgnoreCase("org.jsoup.nodes.Attributes$1");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("V)\"'YcoyNS4}lc4J", "V)\"'YcoyNS4}lc4J");
      boolean boolean0 = attributes1.hasKeyIgnoreCase("V)\"'YcoyNS4}lc4J");
      assertEquals(1, attributes0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.addAll(attributes0);
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put(")\"'YcoyNS4}lc4J", ")\"'YcoyNS4}lc4J");
      attributes1.asList();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("Lz%X4Gr+,ILGUeyE", true);
      List<Attribute> list0 = attributes1.asList();
      assertEquals(1, attributes0.size());
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("}m^s}qs!w", true);
      attributes0.toString();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("zG)#G-VMjdE><H^M~", true);
      StringWriter stringWriter0 = new StringWriter();
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      attributes0.html((Appendable) stringWriter0, document_OutputSettings0);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put(")\"'YcoyNS4}lc4J", ")\"'YcoyNS4}lc4J");
      String string0 = attributes1.toString();
      assertEquals(1, attributes0.size());
      assertEquals(" )\"'YcoyNS4}lc4J=\")&quot;'YcoyNS4}lc4J\"", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals(attributes0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Object object0 = new Object();
      boolean boolean0 = attributes0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put(")\"'YcoyNS4}lc4J", ")\"'YcoyNS4}lc4J");
      Attributes attributes2 = attributes1.clone();
      Attributes attributes3 = attributes2.put(")\"'YcoyNS4}lc4J", false);
      attributes3.equals(attributes0);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("details", "iw\"QCffIZ0HXo0BP", attributes0);
      Attributes attributes1 = attributes0.put(attribute0);
      Attributes attributes2 = attributes0.clone();
      boolean boolean0 = attributes2.equals(attributes1);
      assertEquals(1, attributes0.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase("Awdu%Ku([Es", "Awdu%Ku([Es");
      attributes0.normalize();
      assertEquals(1, attributes0.size());
  }
}