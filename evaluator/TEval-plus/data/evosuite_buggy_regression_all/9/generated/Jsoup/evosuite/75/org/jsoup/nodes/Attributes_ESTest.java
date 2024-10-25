/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:15:05 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
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
      assertTrue(map0.isEmpty());
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
      boolean boolean0 = attributes0.equals(attributes1);
      assertTrue(boolean0);
      assertNotSame(attributes1, attributes0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.putIgnoreCase("8#", "8#");
      Attributes attributes1 = attributes0.put("ZH@Dg(,&~jBg?+w", "ZH@Dg(,&~jBg?+w");
      Attributes attributes2 = attributes1.put("E'$>", "8#");
      attributes1.addAll(attributes2);
      assertEquals(3, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("E'$>", "E'$>");
      attributes0.put("~TZ1&Uj|FF", " E'$>=\"E'$>\"");
      Attributes attributes2 = attributes1.put(" E'$>=\"E'$>\"", true);
      Attributes attributes3 = attributes1.clone();
      attributes1.addAll(attributes2);
      assertEquals(4, attributes0.size());
      assertNotSame(attributes0, attributes3);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("TH>`tZZ", "TH>`tZZ");
      String string0 = attributes1.get("TH>`tZZ");
      assertEquals(1, attributes0.size());
      assertEquals("TH>`tZZ", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      String string0 = Attributes.checkNotNull((String) null);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.get(" o.jsoup.nodes.Attributes$1=\"o.jsoup.nodes.Attributes$1\"");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("org.jsoup.nodes.Attrbutes$Dataset$EntrySet", "org.jsoup.nodes.Attrbutes$Dataset$EntrySet");
      attributes0.getIgnoreCase("org.jsoup.nodes.Attrbutes$Dataset$EntrySet");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      String string0 = attributes0.getIgnoreCase("HO-Nyz=B`");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("MQ<+um,A5;R&F3F", "MQ<+um,A5;R&F3F");
      attributes1.putIgnoreCase("MQ<+um,A5;R&F3F", "(-h5xtzd3ir2");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("MQ<+um,A5;R&F3F", "MQ<+um,A5;R&F3F");
      attributes1.normalize();
      attributes1.putIgnoreCase("MQ<+um,A5;R&F3F", "(-h5xtzd3ir2");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("org.jsoup.nodes.Attributes$Dataset$EntrySet", false);
      assertEquals(0, attributes1.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("ZH@Dg(,&~jBg?+w", "ZH@Dg(,&~jBg?+w");
      attributes1.put("E'$>", "8#");
      attributes1.removeIgnoreCase("ZH@Dg(,&~jBg?+w");
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("org.jsoup.nodes.Attributes$Dataset$EntrySet", "org.jsoup.nodes.Attributes$Dataset$EntrySet");
      assertEquals(1, attributes0.size());
      
      attributes0.put("org.jsoup.nodes.Attributes$Dataset$EntrySet", false);
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.removeIgnoreCase("{ md_B-zpj*K8");
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKey("[H!|/3m}Yn5<y<");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("{k&IU{VLGv", "{k&IU{VLGv");
      boolean boolean0 = attributes1.hasKey("{k&IU{VLGv");
      assertEquals(1, attributes0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.hasKeyIgnoreCase("bdo");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("object", "object");
      boolean boolean0 = attributes1.hasKeyIgnoreCase("object");
      assertEquals(1, attributes0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.addAll(attributes0);
      assertEquals(0, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put(":3%)Ls]ue f", true);
      attributes0.asList();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("Ergujsoup.nodes.Attributes$Dataset$EntrySet", "Ergujsoup.nodes.Attributes$Dataset$EntrySet");
      attributes1.asList();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("E'$>", "E'$>");
      attributes0.html();
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("Ka;mbEVo", (String) null);
      MockFile mockFile0 = new MockFile("$", "$");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(mockFileOutputStream0);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      attributes0.html((Appendable) mockPrintWriter0, document_OutputSettings1);
      assertEquals(1, attributes0.size());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("6-HL`M01Z", true);
      String string0 = attributes1.html();
      assertEquals(1, attributes0.size());
      assertEquals(" 6-HL`M01Z", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.put("|", "org.jsoup.nodes.Attributes$Dataset$EntrySet");
      String string0 = attributes1.html();
      assertEquals(1, attributes0.size());
      assertEquals(" |=\"org.jsoup.nodes.Attributes$Dataset$EntrySet\"", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("typemustmatch", "typemustmatch");
      attributes0.html();
      assertEquals(1, attributes0.size());
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
      boolean boolean0 = attributes0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      boolean boolean0 = attributes0.equals("Y4afOTzC%5:sE!&[\"");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.clone();
      assertEquals(0, attributes1.size());
      
      attributes1.putIgnoreCase("noembed", "noembed");
      boolean boolean0 = attributes0.equals(attributes1);
      assertEquals(1, attributes1.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = attributes0.clone();
      attributes0.put("{`xgj(`jjl]vg{;l1#<", "VgG3t-N73^");
      attributes1.putIgnoreCase("oembed", "oembed");
      boolean boolean0 = attributes0.equals(attributes1);
      assertEquals(1, attributes1.size());
      assertFalse(boolean0);
  }
}
