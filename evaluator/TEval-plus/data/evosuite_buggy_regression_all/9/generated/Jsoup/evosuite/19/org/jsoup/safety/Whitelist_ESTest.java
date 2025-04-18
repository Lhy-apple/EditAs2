/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:05:08 GMT 2023
 */

package org.jsoup.safety;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attribute;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.jsoup.safety.Whitelist;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Whitelist_ESTest extends Whitelist_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.none();
      boolean boolean0 = whitelist0.isSafeTag("ZRF.N");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basicWithImages();
      Whitelist whitelist1 = whitelist0.addEnforcedAttribute("a", "a", "a");
      assertSame(whitelist0, whitelist1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "b";
      Whitelist whitelist1 = whitelist0.addAttributes("blockquote", stringArray0);
      assertSame(whitelist1, whitelist0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("b", "b");
      Whitelist whitelist0 = Whitelist.simpleText();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "b";
      whitelist0.addAttributes("b", stringArray0);
      whitelist0.addProtocols("b", "b", stringArray0);
      Document document0 = Document.createShell("b");
      boolean boolean0 = whitelist0.isSafeAttribute("b", document0, attribute0);
      assertEquals("", attribute0.getValue());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basicWithImages();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "b";
      Whitelist whitelist1 = whitelist0.addProtocols("b", "b", stringArray0);
      Whitelist whitelist2 = whitelist0.addProtocols("b", "b", stringArray0);
      assertSame(whitelist2, whitelist1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      boolean boolean0 = whitelist0.isSafeTag("b");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = Document.createShell(".<-'Jbs");
      Attribute attribute0 = new Attribute("b", ".<-'Jbs");
      Whitelist whitelist0 = Whitelist.basicWithImages();
      boolean boolean0 = whitelist0.isSafeAttribute("pm", document0, attribute0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document(".xJbBs");
      Attribute attribute0 = new Attribute(".xJbBs", ".xJbBs");
      Whitelist whitelist0 = Whitelist.basicWithImages();
      boolean boolean0 = whitelist0.isSafeAttribute("a", document0, attribute0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = Document.createShell(".<-'Jbs");
      Attribute attribute0 = new Attribute("b", ".<-'Jbs");
      Whitelist whitelist0 = Whitelist.basicWithImages();
      String[] stringArray0 = new String[3];
      stringArray0[0] = ".<-'Jbs";
      stringArray0[1] = "b";
      stringArray0[2] = ".<-'Jbs";
      Whitelist whitelist1 = whitelist0.addAttributes(":all", stringArray0);
      boolean boolean0 = whitelist1.isSafeAttribute("pm", document0, attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("b", "b");
      Whitelist whitelist0 = Whitelist.basic();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "b";
      whitelist0.addAttributes("b", stringArray0);
      Whitelist whitelist1 = whitelist0.addProtocols("b", "img", stringArray0);
      Document document0 = new Document("b");
      boolean boolean0 = whitelist1.isSafeAttribute("b", document0, attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("b", "b");
      Whitelist whitelist0 = Whitelist.simpleText();
      String[] stringArray0 = new String[1];
      whitelist0.preserveRelativeLinks(true);
      stringArray0[0] = "b";
      whitelist0.addAttributes("b", stringArray0);
      whitelist0.addProtocols("b", "b", stringArray0);
      Document document0 = Document.createShell("b");
      boolean boolean0 = whitelist0.isSafeAttribute("b", document0, attribute0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basicWithImages();
      Attributes attributes0 = whitelist0.getEnforcedAttributes(".'JbBs");
      assertNotNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.simpleText();
      whitelist0.addEnforcedAttribute("b", "b", "b");
      Attributes attributes0 = whitelist0.getEnforcedAttributes("b");
      assertNotNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Whitelist.AttributeValue whitelist_AttributeValue0 = Whitelist.AttributeValue.valueOf("org.jsoup.safety.Whitelist$TypedValue");
      boolean boolean0 = whitelist_AttributeValue0.equals(whitelist_AttributeValue0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Whitelist.AttributeValue whitelist_AttributeValue0 = new Whitelist.AttributeValue("org.jsoup.safety.Whitelist$TypedValue");
      boolean boolean0 = whitelist_AttributeValue0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Whitelist.TagName whitelist_TagName0 = Whitelist.TagName.valueOf(":W(fhAkkIo");
      boolean boolean0 = whitelist_TagName0.equals(":W(fhAkkIo");
      assertFalse(boolean0);
  }
}
