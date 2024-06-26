/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:49:26 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedHashSet;
import java.util.NoSuchElementException;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("select");
      document0.appendElement("select");
      String string0 = document0.outerHtml();
      assertEquals("<select></select>", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("#root");
      document0.appendText("");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document(";X");
      Element element0 = document0.val(";X");
      assertEquals("#root", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("h^z)");
      Elements elements0 = document0.getElementsByAttributeValueEnding("h^z)", "h^z)");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("orY.jsoup.select.Selecor");
      Elements elements0 = document0.select("orY.jsoup.select.Selecor");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("SRG");
      // Undeclared exception!
      try { 
        document0.child(817);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 817, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("&C`");
      Elements elements0 = document0.getElementsByAttributeValueContaining("&C`", "&C`");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "textarea", attributes0);
      Element element1 = element0.val("org.jsoup.nodes.Element");
      assertFalse(element1.isBlock());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("orY.jsoup.select.Selecor");
      Elements elements0 = document0.getElementsByAttributeValueNot("orY.jsoup.select.Selecor", "orY.jsoup.select.Selecor");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("STRENG");
      String string0 = document0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("}h^z)");
      Elements elements0 = document0.getElementsByAttribute("}h^z)");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("(a}q;Sf[}a~RE?Mo w");
      document0.addClass("(a}q;Sf[}a~RE?Mo w");
      Element element0 = document0.toggleClass("(a}q;Sf[}a~RE?Mo w");
      assertEquals("#document", element0.nodeName());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("orY.jsoup.select.Selecor");
      // Undeclared exception!
      try { 
        document0.html("orY.jsoup.select.Selecor");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("org.jsoup.selec.Selector");
      Elements elements0 = document0.getElementsByAttributeValue("org.jsoup.selec.Selector", "vaue");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("STRNG");
      // Undeclared exception!
      try { 
        document0.siblingElements();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Element", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document("h^z)");
      Elements elements0 = document0.getElementsByAttributeValueStarting("h^z)", "h^z)");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Kw&L1T-4QJ8|y");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "Kw&L1T-4QJ8|y", attributes0);
      String string0 = element0.nodeName();
      assertEquals("kw&l1t-4qj8|y", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("dsv]{.Cz!Z");
      Element element0 = document0.removeClass("dsv]{.Cz!Z");
      assertEquals("dsv]{.Cz!Z", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("STRNG");
      Elements elements0 = document0.getAllElements();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("ST_RNG");
      Elements elements0 = document0.getElementsByIndexLessThan(25);
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("B;<NF4ClG$z");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1717986954));
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("class");
      Elements elements0 = document0.getElementsByIndexEquals(13);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("B^:");
      Element element0 = document0.appendElement("B^:");
      Elements elements0 = element0.parents();
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Tag tag0 = Tag.valueOf("jsoup");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "jsoup", attributes0);
      Element element1 = element0.appendElement(".Rt. *BV@T>;YJ(/");
      Elements elements0 = element1.parents();
      assertEquals(".rt. *bv@t>;yj(/", element1.nodeName());
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Selector");
      document0.prependChild(document0);
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("B;<hF4q8$z");
      Element element0 = document0.appendElement("B;<hF4q8$z");
      document0.prependText("B;<hF4q8$z");
      Element element1 = element0.lastElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("STRONG");
      Element element0 = document0.appendElement("STRONG");
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("ewop$hz");
      document0.appendChild(document0);
      document0.appendElement("ewop$hz");
      Element element0 = document0.nextElementSibling();
      assertEquals("ewop$hz", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Selector");
      Element element0 = document0.prependChild(document0);
      Element element1 = element0.appendElement("u");
      Element element2 = element1.previousElementSibling();
      assertNotNull(element2);
      assertEquals("u", element1.tagName());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("STRONG");
      Element element0 = document0.appendElement("STRONG");
      Element element1 = element0.firstElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("STONG");
      document0.appendChild(document0);
      Element element0 = document0.appendElement("STONG");
      Element element1 = element0.firstElementSibling();
      assertEquals("#document", element1.nodeName());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("STRNG");
      Element element0 = document0.appendElement("STRNG");
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("STRNG");
      document0.addChild(document0);
      Element element0 = document0.appendElement("STRNG");
      Element element1 = element0.lastElementSibling();
      assertEquals("strng", element1.nodeName());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document(":~ s#-jFYX-orG%w;");
      Element element0 = document0.getElementById(":~ s#-jFYX-orG%w;");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("textarea");
      document0.prependText("s+");
      String string0 = document0.text();
      assertEquals("s+", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.appendElement("textarea");
      element0.prependText("s+");
      String string0 = document0.text();
      assertEquals("s+", string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.appendElement("textarea");
      Element element1 = element0.prependChild(document0);
      element1.prependText("s+");
      // Undeclared exception!
      try { 
        element1.text();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document("class");
      Element element0 = document0.appendElement("class");
      element0.prependText("class");
      String string0 = document0.outerHtml();
      assertEquals("<class>\nclass\n</class>", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Tag tag0 = Tag.valueOf("jsoup");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "jsoup", attributes0);
      Element element1 = element0.appendElement(".Rt. *BV@T>;YJ(/");
      assertEquals(".rt. *bv@t>;yj(/", element1.nodeName());
      
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("STRONG");
      Element element0 = document0.appendElement("STRONG");
      element0.text("STRONG");
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("4_D?xn97KsNH*.nCIly");
      DataNode dataNode0 = new DataNode("4_D?xn97KsNH*.nCIly", "4_D?xn97KsNH*.nCIly");
      document0.appendChild(dataNode0);
      String string0 = document0.data();
      assertEquals("4_D?xn97KsNH*.nCIly", string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("4_D?xn97KsNH*.nCIly");
      document0.appendElement("4_D?xn97KsNH*.nCIly");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("~,5xhj:qEq)/Z%3ZBQA");
      Element element0 = document0.prependText("~,5xhj:qEq)/Z%3ZBQA");
      String string0 = element0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("orY.jsoup.select.Selecor");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      document0.classNames((Set<String>) linkedHashSet0);
      Set<String> set0 = document0.classNames();
      assertFalse(set0.contains("orY.jsoup.select.Selecor"));
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document(":~ s#-jFYX-orG%w;");
      boolean boolean0 = document0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document("STRN");
      Element element0 = document0.toggleClass("STRN");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("B;<NF4C8G$z");
      String string0 = document0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Element element0 = new Element(tag0, " />");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("STRONG");
      StringBuilder stringBuilder0 = new StringBuilder();
      document0.outerHtml(stringBuilder0);
      assertEquals("<#root>\n</#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Element element0 = new Element(tag0, "4*+{}(4S");
      Element element1 = element0.appendChild(element0);
      // Undeclared exception!
      try { 
        element1.outerHtml((StringBuilder) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Element", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Selecor");
      document0.prependElement("org.jsoup.select.Selecor");
      Element element0 = document0.prependText("org.jsoup.select.Selecor");
      String string0 = ((Document) element0).outerHtml();
      assertEquals("org.jsoup.select.Selecor<org.jsoup.select.selecor>\n</org.jsoup.select.selecor>", string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("h^z)");
      boolean boolean0 = document0.equals("h^z)");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document(";X");
      // Undeclared exception!
      try { 
        document0.normalise();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }
}
