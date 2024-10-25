/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:18:08 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
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
      Attributes attributes1 = attributes0.put("T  ", "T  ");
      attributes1.addAll(attributes0);
      attributes1.removeIgnoreCase("T  ");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Map<String, String> map0 = attributes0.dataset();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.hashCode();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.clone();
      boolean boolean0 = attributes1.equals(attributes0);
      assertNotSame(attributes1, attributes0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("data-", "4E%6K?`@cTTVdwot+Wp");
      Attributes attributes2 = attributes1.put("hD/`sf.)T4", true);
      Attribute attribute0 = Attribute.createFromEncoded("truespeed", "truespeed");
      attributes0.put(attribute0);
      attributes0.addAll(attributes2);
      assertEquals(3, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("4E%6K?`@cTTVdwot+Wp", "4E%6K?`@cTTVdwot+Wp");
      Attribute attribute0 = Attribute.createFromEncoded("truespeed", "truespeed");
      attributes0.put(attribute0);
      attributes0.put("org.jsoup.nodes.Attributes$1", "{Y?T^wgfV~Ar9");
      Attributes attributes1 = attributes0.clone();
      attributes0.addAll(attributes1);
      assertEquals(3, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("T  ", "T  ");
      assertEquals(1, attributes0.size());
      
      attributes1.removeIgnoreCase("T  ");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("zzyt", "zzyt");
      String string0 = attributes1.getIgnoreCase("zzyt");
      assertEquals(1, attributes0.size());
      assertEquals("zzyt", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String string0 = Attributes.checkNotNull((String) null);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("Children collection to [e inserted must not be null.", "Children collection to [e inserted must not be null.");
      String string0 = attributes1.get("Children collection to [e inserted must not be null.");
      assertEquals(1, attributes0.size());
      assertEquals("Children collection to [e inserted must not be null.", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.get("Children collection to [e inserted must not be null.");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.getIgnoreCase("G>8#U[r6");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("org.jsoup.nodes.Attributes$Dataset$EntrySet", "org.jsoup.nodes.Attributes$Dataset$EntrySet");
      attributes1.put("org.jsoup.nodes.Attributes$Dataset$EntrySet", true);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("org.jsoup.nodes.Attributes$Dataset$EntrySet", "org.jsoup.nodes.Attributes$Dataset$EntrySet");
      attributes1.normalize();
      attributes1.put("org.jsoup.nodes.Attributes$Dataset$EntrySet", true);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("*yBmb5", false);
      assertEquals(0, attributes1.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("f#+/PKyb,", "f#+/PKyb,");
      assertEquals(1, attributes0.size());
      
      attributes1.remove("f#+/PKyb,");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.removeIgnoreCase(">zzt>n=l,[");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKey("UG*!Axm~2A");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("4E%6K?`@cTTVdwot+Wp", true);
      boolean boolean0 = attributes0.hasKey("4E%6K?`@cTTVdwot+Wp");
      assertEquals(1, attributes0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKeyIgnoreCase("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("org.jsoup.select.evaluator$indexlessthan", "org.jsoup.select.evaluator$indexlessthan");
      boolean boolean0 = attributes1.hasKeyIgnoreCase("org.jsoup.select.evaluator$indexlessthan");
      assertEquals(1, attributes0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.addAll(attributes0);
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase("org.jsoup.nodes.Attributes$Dataset$EntrySet", "org.jsoup.nodes.Attributes$Dataset$EntrySet");
      attributes0.asList();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase("org.jsoup.nodes.Attributes$Dataset$EntrySet", (String) null);
      attributes0.asList();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("Children collection to [e inserted must not be null.", "Children collection to [e inserted must not be null.");
      String string0 = attributes1.html();
      assertEquals(1, attributes0.size());
      assertEquals(" Children collection to [e inserted must not be null.=\"Children collection to [e inserted must not be null.\"", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("eU3nH<%ySKE=r@", (String) null);
      attributes0.html();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("Children collection to [e inserted must not be null.", "menuitem");
      String string0 = attributes0.html();
      assertEquals(" Children collection to [e inserted must not be null.=\"menuitem\"", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("readonly", "readonly");
      attributes0.html();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("@dJw~=rD;~", (String) null);
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream(pipedInputStream0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(pipedOutputStream0, false);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      attributes0.html((Appendable) mockPrintStream0, document_OutputSettings0);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals(attributes0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals("org.jsoup.nodes.Attributes$Dataset$DatasetIterator");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = new Attributes();
      attributes1.put("&#xa0;", "&#xa0;");
      attributes0.equals(attributes1);
      assertEquals(1, attributes1.size());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("4E%6K?`@cTTVdwot+Wp", true);
      Attributes attributes2 = attributes1.clone();
      boolean boolean0 = attributes2.equals(attributes0);
      assertEquals(1, attributes0.size());
      assertFalse(boolean0);
  }
}
