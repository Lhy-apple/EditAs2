/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:49:10 GMT 2023
 */

package org.jsoup.safety;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
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
      Whitelist whitelist0 = Whitelist.simpleText();
      boolean boolean0 = whitelist0.isSafeTag("d?Q0t4%U35:kwC>V");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basic();
      whitelist0.preserveRelativeLinks(true);
      String[] stringArray0 = new String[1];
      stringArray0[0] = "cite";
      Whitelist whitelist1 = whitelist0.addAttributes("cite", stringArray0);
      Attribute attribute0 = new Attribute("cite", "cite");
      Document document0 = Document.createShell("cite");
      boolean boolean0 = whitelist1.isSafeAttribute("cite", document0, attribute0);
      assertEquals("cite", attribute0.getValue());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.none();
      assertNotNull(whitelist0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basicWithImages();
      Attributes attributes0 = whitelist0.getEnforcedAttributes("a");
      assertNotNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "blockquote";
      Whitelist whitelist1 = whitelist0.addAttributes("blockquote", stringArray0);
      Attribute attribute0 = Attribute.createFromEncoded("blockquote", "blockquote");
      Document document0 = Document.createShell("blockquote");
      boolean boolean0 = whitelist1.isSafeAttribute("blockquote", document0, attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basic();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "cite";
      Whitelist whitelist1 = whitelist0.addAttributes("cite", stringArray0);
      Attribute attribute0 = new Attribute("cite", "cite");
      Document document0 = Document.createShell("cite");
      boolean boolean0 = whitelist1.isSafeAttribute("cite", document0, attribute0);
      assertEquals("", attribute0.getValue());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      whitelist0.addEnforcedAttribute("clL2\"[m;91L_MT6F,4", "clL2\"[m;91L_MT6F,4", "clL2\"[m;91L_MT6F,4");
      Whitelist whitelist1 = whitelist0.addEnforcedAttribute("clL2\"[m;91L_MT6F,4", "clL2\"[m;91L_MT6F,4", "clL2\"[m;91L_MT6F,4");
      assertSame(whitelist1, whitelist0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basic();
      String[] stringArray0 = new String[1];
      // Undeclared exception!
      try { 
        whitelist0.addProtocols("cite", "cite", stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      boolean boolean0 = whitelist0.isSafeTag("dt");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basic();
      Attribute attribute0 = Attribute.createFromEncoded("cite", ":all");
      Document document0 = Document.createShell("cite");
      boolean boolean0 = whitelist0.isSafeAttribute("cite", document0, attribute0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      Attribute attribute0 = new Attribute("i\tpro", "i\tpro");
      Document document0 = new Document("i\tpro");
      boolean boolean0 = whitelist0.isSafeAttribute("ol", document0, attribute0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basic();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "cite";
      whitelist0.addAttributes(":all", stringArray0);
      Attribute attribute0 = Attribute.createFromEncoded("cite", ":all");
      Document document0 = Document.createShell("cite");
      boolean boolean0 = whitelist0.isSafeAttribute("cite", document0, attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.basic();
      Attributes attributes0 = whitelist0.getEnforcedAttributes("sr");
      assertNotNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Whitelist.AttributeKey whitelist_AttributeKey0 = Whitelist.AttributeKey.valueOf("7UWYe0K74I4");
      boolean boolean0 = whitelist_AttributeKey0.equals(whitelist_AttributeKey0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Whitelist.AttributeKey whitelist_AttributeKey0 = new Whitelist.AttributeKey("dd");
      boolean boolean0 = whitelist_AttributeKey0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Whitelist.AttributeKey whitelist_AttributeKey0 = Whitelist.AttributeKey.valueOf("a");
      boolean boolean0 = whitelist_AttributeKey0.equals("a");
      assertFalse(boolean0);
  }
}
