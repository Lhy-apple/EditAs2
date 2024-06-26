/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:38:20 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringWriter;
import java.nio.CharBuffer;
import java.util.Map;
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
      String string0 = attributes0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Map<String, String> map0 = attributes0.dataset();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.hashCode();
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
      Attribute attribute0 = new Attribute("V1Y,XB9a$V*O[b#py`C", "V1Y,XB9a$V*O[b#py`C");
      Attributes attributes1 = attributes0.put(attribute0);
      Attributes attributes2 = attributes0.put(" V1Y,XB9a$V*O[b#py`C=\"V1Y,XB9a$V*O[b#py`C\"", true);
      attributes2.addAll(attributes2);
      attributes1.addAll(attributes1);
      assertEquals(3, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("V1Y,XB9a$V*O[b#py`C", "V1Y,XB9a$V*O[b#py`C");
      Attributes attributes1 = attributes0.put(attribute0);
      Attributes attributes2 = attributes1.put(" V1Y,XB9a$V*O[b#py`C=\"V1Y,XB9a$V*O[b#py`C\"", true);
      Attributes attributes3 = attributes0.clone();
      Attributes attributes4 = attributes3.put(attribute0);
      attributes4.addAll(attributes1);
      attributes2.addAll(attributes4);
      assertEquals(3, attributes0.size());
      assertFalse(attributes0.equals((Object)attributes3));
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", true);
      String string0 = attributes1.getIgnoreCase("");
      assertEquals(1, attributes0.size());
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("V1Y,XB9a$V*O[b#py`C", "V1Y,XB9a$V*O[b#py`C");
      Attributes attributes1 = attributes0.put(attribute0);
      String string0 = attributes1.get("V1Y,XB9a$V*O[b#py`C");
      assertEquals(1, attributes0.size());
      assertEquals("V1Y,XB9a$V*O[b#py`C", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.get("CJS|~*AE/uve0iwhu:");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.getIgnoreCase("uT7S*S>yqNmGQJgVm#p");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("fe~fJPl<s[{6=G", "fe~fJPl<s[{6=G");
      attributes1.put("fe~fJPl<s[{6=G", true);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("fe~fJPl<s[{6=G", "fe~fJPl<s[{6=G");
      attributes0.normalize();
      attributes1.put("fe~fJPl<s[{6=G", true);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("h", false);
      assertEquals(0, attributes1.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put(" V1Y,XB9a$V*O[b#py`C=\"V1Y,XB9a$V*O[b#py`C\"", true);
      attributes1.addAll(attributes1);
      attributes1.removeIgnoreCase(" V1Y,XB9a$V*O[b#py`C=\"V1Y,XB9a$V*O[b#py`C\"");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("!FpX\"", "!FpX\"");
      assertEquals(1, attributes0.size());
      
      attributes1.remove("!FpX\"");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.removeIgnoreCase("Y");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKey("j");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("j", "j");
      boolean boolean0 = attributes1.hasKey("j");
      assertEquals(1, attributes0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKeyIgnoreCase("j$?$)&|52eq");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", "");
      boolean boolean0 = attributes1.hasKeyIgnoreCase("");
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
      Attributes attributes1 = attributes0.put("NT\"+?PB6qS0A(l_1{", "NT\"+?PB6qS0A(l_1{");
      attributes1.asList();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", true);
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
  public void test23()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("V1Y,XB9a$V*O[b#py`C", "V1Y,XB9a$V*O[b#py`C");
      Attributes attributes1 = attributes0.put(attribute0);
      String string0 = attributes1.html();
      assertEquals(" V1Y,XB9a$V*O[b#py`C=\"V1Y,XB9a$V*O[b#py`C\"", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("", true);
      String string0 = attributes1.html();
      assertEquals(1, attributes0.size());
      assertEquals(" ", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = Attribute.createFromEncoded("i ", "i ");
      Attributes attributes1 = attributes0.put(attribute0);
      String string0 = attributes1.html();
      assertEquals(1, attributes0.size());
      assertEquals(" i=\"i \"", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase("nohref", "nohref");
      CharBuffer charBuffer0 = CharBuffer.allocate(7);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      attributes0.html((Appendable) charBuffer0, document_OutputSettings0);
      assertEquals(7, charBuffer0.position());
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("org.jsoup.select.Evaluator$AttributeKeyPair", true);
      StringWriter stringWriter0 = new StringWriter(7);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      attributes0.html((Appendable) stringWriter0, document_OutputSettings1);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals(attributes0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals("Nb0K\"@");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.clone();
      attributes1.putIgnoreCase("rtn0f-rfb44f|bc5^yb", "rtn0f-rfb44f|bc5^yb");
      boolean boolean0 = attributes1.equals(attributes0);
      assertEquals(1, attributes1.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("h", "h");
      Attributes attributes2 = attributes1.clone();
      boolean boolean0 = attributes2.equals(attributes0);
      assertEquals(1, attributes0.size());
      assertFalse(boolean0);
  }
}
