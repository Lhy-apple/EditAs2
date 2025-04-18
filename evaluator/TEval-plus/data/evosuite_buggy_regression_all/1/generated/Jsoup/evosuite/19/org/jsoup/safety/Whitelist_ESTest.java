/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:47:16 GMT 2023
 */

package org.jsoup.safety;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attribute;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.safety.Whitelist;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Whitelist_ESTest extends Whitelist_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.simpleText();
      Document document0 = Document.createShell("]");
      String[] stringArray0 = new String[1];
      stringArray0[0] = "]";
      Whitelist whitelist1 = whitelist0.addAttributes("]", stringArray0);
      Attribute attribute0 = new Attribute("]", "]");
      whitelist1.addProtocols("]", "]", stringArray0);
      boolean boolean0 = whitelist1.isSafeAttribute("]", document0, attribute0);
      assertEquals("", attribute0.getValue());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basicWithImages();
      Attributes attributes0 = whitelist0.getEnforcedAttributes("}_,BgYWuPTZ:");
      assertNotNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "col";
      stringArray0[1] = "col";
      Whitelist whitelist1 = whitelist0.addAttributes("col", stringArray0);
      assertSame(whitelist1, whitelist0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Whitelist whitelist0 = new Whitelist();
      Whitelist whitelist1 = whitelist0.addEnforcedAttribute("4F*^bpT\"{ ", "4F*^bpT\"{ ", "4F*^bpT\"{ ");
      Whitelist whitelist2 = whitelist1.addEnforcedAttribute("4F*^bpT\"{ ", "4F*^bpT\"{ ", "4F*^bpT\"{ ");
      assertSame(whitelist2, whitelist0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Whitelist whitelist0 = new Whitelist();
      String[] stringArray0 = new String[5];
      stringArray0[0] = "j10lt_Nkg";
      stringArray0[1] = "j10lt_Nkg";
      stringArray0[2] = "j10lt_Nkg";
      stringArray0[3] = "j10lt_Nkg";
      stringArray0[4] = "j10lt_Nkg";
      Whitelist whitelist1 = whitelist0.addProtocols("j10lt_Nkg", "j10lt_Nkg", stringArray0);
      Whitelist whitelist2 = whitelist1.addProtocols("j10lt_Nkg", "j10lt_Nkg", stringArray0);
      assertSame(whitelist2, whitelist0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.none();
      boolean boolean0 = whitelist0.isSafeTag("$UtRi$BA");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      boolean boolean0 = whitelist0.isSafeTag("sup");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.simpleText();
      Document document0 = Document.createShell("]");
      Attribute attribute0 = Attribute.createFromEncoded("]", ":all");
      boolean boolean0 = whitelist0.isSafeAttribute("]", document0, attribute0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      Attribute attribute0 = Attribute.createFromEncoded("ol", "ol");
      boolean boolean0 = whitelist0.isSafeAttribute("ol", (Element) null, attribute0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.simpleText();
      Document document0 = Document.createShell("]");
      String[] stringArray0 = new String[1];
      stringArray0[0] = "]";
      whitelist0.addAttributes(":all", stringArray0);
      Attribute attribute0 = Attribute.createFromEncoded("]", ":all");
      boolean boolean0 = whitelist0.isSafeAttribute("]", document0, attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.simpleText();
      Document document0 = Document.createShell("]");
      String[] stringArray0 = new String[1];
      stringArray0[0] = "]";
      whitelist0.addAttributes("]", stringArray0);
      Attribute attribute0 = Attribute.createFromEncoded("]", "]");
      whitelist0.addProtocols("]", "Tvi4vU1/1S", stringArray0);
      boolean boolean0 = whitelist0.isSafeAttribute("]", document0, attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.none();
      Document document0 = Document.createShell("]");
      String[] stringArray0 = new String[2];
      stringArray0[0] = "]";
      stringArray0[1] = "]";
      Whitelist whitelist1 = whitelist0.addAttributes("]", stringArray0);
      whitelist1.preserveRelativeLinks(true);
      Attribute attribute0 = Attribute.createFromEncoded("]", "]");
      whitelist0.addProtocols("]", "]", stringArray0);
      boolean boolean0 = whitelist1.isSafeAttribute("]", document0, attribute0);
      assertFalse(boolean0);
      assertEquals("]", attribute0.getValue());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      Whitelist whitelist1 = whitelist0.addEnforcedAttribute("[d}HSkoA", "[d}HSkoA", "[d}HSkoA");
      Attributes attributes0 = whitelist1.getEnforcedAttributes("[d}HSkoA");
      assertNotNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Whitelist.AttributeValue whitelist_AttributeValue0 = new Whitelist.AttributeValue("ul");
      boolean boolean0 = whitelist_AttributeValue0.equals(whitelist_AttributeValue0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Whitelist.AttributeKey whitelist_AttributeKey0 = new Whitelist.AttributeKey("sup");
      boolean boolean0 = whitelist_AttributeKey0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Whitelist.AttributeKey whitelist_AttributeKey0 = Whitelist.AttributeKey.valueOf("<`8Tw");
      boolean boolean0 = whitelist_AttributeKey0.equals("<`8Tw");
      assertFalse(boolean0);
  }
}
