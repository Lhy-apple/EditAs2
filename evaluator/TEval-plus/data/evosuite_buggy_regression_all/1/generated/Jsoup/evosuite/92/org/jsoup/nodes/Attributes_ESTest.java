/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:52:25 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.CharArrayWriter;
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
      attributes0.dataset();
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.hashCode();
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.clone();
      Attribute attribute0 = Attribute.createFromEncoded("_l!7gZs", "_l!7gZs");
      Attributes attributes2 = attributes1.put(attribute0);
      attributes2.equals(attributes0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", (String) null);
      Attributes attributes2 = attributes1.put(" {Y@v>KUA?", true);
      Attribute attribute0 = Attribute.createFromEncoded(" {Y@v>KUA?", " {Y@v>KUA?");
      attributes2.put(attribute0);
      // Undeclared exception!
      try { 
        attributes0.addAll(attributes0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("UHtt[uus", true);
      Attributes attributes2 = attributes0.put("b9buPH|X7", true);
      Attributes attributes3 = attributes1.put(">&4=.\"E ", "b9buPH|X7");
      attributes3.clone();
      attributes2.addAll(attributes1);
      assertEquals(4, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", "");
      assertEquals(1, attributes0.size());
      
      attributes1.removeIgnoreCase("");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", "");
      String string0 = attributes1.get("");
      assertEquals(1, attributes0.size());
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.get("4Y%");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put(")\"'YcoyNS4}lc4J", ")\"'YcoyNS4}lc4J");
      String string0 = attributes1.getIgnoreCase(")\"'YcoyNS4}lc4J");
      assertEquals(1, attributes0.size());
      assertEquals(")\"'YcoyNS4}lc4J", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.getIgnoreCase("j}C\"%BTRj}bjq");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put(")\"'YcoyNS4}lc4J", ")\"'YcoyNS4}lc4J");
      attributes0.put(")\"'YcoyNS4}lc4J", true);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put(")\"'YcoyNS4}lc4J", ")\"'YcoyNS4}lc4J");
      attributes1.normalize();
      attributes0.put(")\"'YcoyNS4}lc4J", true);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("Qt4$b?jrfpJ~Q2@Q$", false);
      assertEquals(0, attributes1.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("8CypUMp", "8CypUMp");
      attributes1.put("rgky.k[1-8k+b", "rgky.k[1-8k+b");
      attributes1.remove("8CypUMp");
      assertEquals(1, attributes1.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.removeIgnoreCase("");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKey(";5>z'=@$_oz{yx$}");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("RgKY.k[1-8K+B", "RgKY.k[1-8K+B");
      boolean boolean0 = attributes1.hasKey("RgKY.k[1-8K+B");
      assertEquals(1, attributes0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKeyIgnoreCase("=\"");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("rJ[ )6", "rJ[ )6");
      boolean boolean0 = attributes1.hasKeyIgnoreCase("rJ[ )6");
      assertEquals(1, attributes0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.addAll(attributes0);
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("       ", "       ");
      // Undeclared exception!
      try { 
        attributes1.asList();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("Lz%X4Gr+,ILGUeyE", true);
      attributes1.asList();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("{;X1F#[W@9", true);
      attributes0.toString();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("RgKY.k[1-8K+B", "RgKY.k[1-8K+B");
      String string0 = attributes1.toString();
      assertEquals(1, attributes0.size());
      assertEquals(" RgKY.k[1-8K+B=\"RgKY.k[1-8K+B\"", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase("GY-8M(.JS", (String) null);
      CharArrayWriter charArrayWriter0 = new CharArrayWriter(4071);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      attributes0.html((Appendable) charArrayWriter0, document_OutputSettings1);
      assertEquals(" GY-8M(.JS=\"\"", charArrayWriter0.toString());
      assertEquals(13, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals(attributes0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Object object0 = new Object();
      boolean boolean0 = attributes0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("_l!7gZs", true);
      Attributes attributes2 = attributes1.clone();
      Attribute attribute0 = Attribute.createFromEncoded("_l!7gZs", "_l!7gZs");
      Attributes attributes3 = attributes2.put(attribute0);
      boolean boolean0 = attributes3.equals(attributes0);
      assertEquals(1, attributes0.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.clone();
      boolean boolean0 = attributes1.equals(attributes0);
      assertNotSame(attributes1, attributes0);
      assertTrue(boolean0);
  }
}