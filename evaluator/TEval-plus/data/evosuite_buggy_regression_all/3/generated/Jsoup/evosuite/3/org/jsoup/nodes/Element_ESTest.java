/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:47:30 GMT 2023
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
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.XmlDeclaration;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("class");
      Element element0 = document0.appendText("class");
      assertEquals("class", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("kK&{!>^_YpF)rSA.");
      Element element0 = document0.val("kK&{!>^_YpF)rSA.");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("7d");
      Element element0 = document0.prependText("7d");
      Elements elements0 = element0.select("7d");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("7d");
      Elements elements0 = document0.getElementsByAttributeValueEnding("7d", "7d");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("855");
      Elements elements0 = document0.select("org.jsoup.nodes.Element");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("=oEmQ\"nU~");
      // Undeclared exception!
      try { 
        document0.child((-1));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("rArr");
      Elements elements0 = document0.getElementsByAttributeValueContaining("rArr", "rArr");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "250", attributes0);
      // Undeclared exception!
      try { 
        element0.val((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The validated object is null
         //
         verifyException("org.apache.commons.lang.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document(",d");
      // Undeclared exception!
      try { 
        document0.html(",d");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("^`f3");
      Elements elements0 = document0.getElementsByAttributeValueNot("^`f3", "^`f3");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document(",d");
      Elements elements0 = document0.getElementsByAttribute(",d");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("Y'UHW^CMEY9m X?W9Q");
      Element element0 = document0.addClass("Y'UHW^CMEY9m X?W9Q");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("^`f3");
      Elements elements0 = document0.getElementsByAttributeValue("^`f3", "^`f3");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document(",d");
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
  public void test14()  throws Throwable  {
      Document document0 = new Document("[P");
      Elements elements0 = document0.getElementsByAttributeValueStarting("[P", "[P");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Tag tag0 = Tag.valueOf("b^8-gky\"m{>8nSVD!N");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "#$W8~6Zh{WjAVu", attributes0);
      String string0 = element0.nodeName();
      assertEquals("b^8-gky\"m{>8nsvd!n", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document(",d");
      Element element0 = document0.removeClass(",d");
      assertEquals("#root", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("7d");
      Elements elements0 = document0.getAllElements();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("EPL");
      Elements elements0 = document0.getElementsByIndexLessThan((-1));
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("3yF)|1");
      document0.prependElement("3yF)|1");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("p3yF)|1");
      Elements elements0 = document0.getElementsByIndexGreaterThan(1187);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("3|tCyhx?K8HY7K");
      Elements elements0 = document0.getElementsByIndexEquals(1);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("Evaluator");
      document0.setParentNode(document0);
      Elements elements0 = document0.parents();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("Evaluator");
      Element element0 = document0.createElement("Fmh4");
      document0.setParentNode(element0);
      Elements elements0 = document0.parents();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("kK&{!>^_YF)rSA.");
      Element element0 = document0.appendChild(document0);
      element0.getElementsByTag("C)7");
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("Y'UHW^CMEY9m X?W9Q");
      document0.appendChild(document0);
      Element element0 = document0.nextElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("Y'UHW^CMEY9m X?W9Q");
      Element element0 = document0.appendChild(document0);
      element0.appendElement("Y'UHW^CMEY9m X?W9Q");
      Element element1 = document0.nextElementSibling();
      assertEquals("y'uhw^cmey9m x?w9q", element1.nodeName());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("yM`poHJ9JW5;NT");
      document0.appendChild(document0);
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("yM`poHJ9JW5;NT");
      document0.appendElement("yM`poHJ9JW5;NT");
      document0.appendChild(document0);
      Element element0 = document0.previousElementSibling();
      assertEquals("yM`poHJ9JW5;NT", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("Y'UHW^CMEY9m X?W9Q");
      document0.appendChild(document0);
      Element element0 = document0.firstElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("Jqh*>P<OW");
      Document document1 = new Document("Jqh*>P<OW");
      Element element0 = document0.appendChild(document1);
      document0.appendChild(element0);
      Element element1 = document0.firstElementSibling();
      assertNotNull(element1);
      assertSame(element1, document1);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.appendChild(document0);
      element0.getElementsByIndexEquals(46);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("");
      document0.appendChild(document0);
      Element element0 = document0.lastElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("");
      document0.appendElement("#`b:u*e");
      document0.appendChild(document0);
      Element element0 = document0.lastElementSibling();
      assertEquals("", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("#root");
      Elements elements0 = document0.select("#root");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("kK&{!>^_YF)rSA.");
      Element element0 = document0.prependElement("textarea");
      Element element1 = document0.prependText("");
      element0.appendChild(element1);
      // Undeclared exception!
      try { 
        document0.text();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("7d");
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("3~B,-tiRvq>5", "3~B,-tiRvq>5", true);
      document0.addChild(xmlDeclaration0);
      String string0 = document0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document("7d");
      Element element0 = document0.prependText("7d");
      element0.appendElement("7d");
      String string0 = document0.text();
      assertEquals("7d", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("7d");
      Element element0 = document0.appendElement("7d");
      boolean boolean0 = element0.preserveWhitespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document(",d");
      document0.appendElement(",d");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("H1");
      Element element0 = document0.prependText("");
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("7");
      Comment comment0 = new Comment("7", "7");
      Element element0 = document0.appendChild(comment0);
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document(",d");
      Element element0 = document0.appendElement(",d");
      element0.prependText(",d");
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("7d");
      Element element0 = document0.prependText("7d");
      String string0 = element0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      document0.classNames((Set<String>) linkedHashSet0);
      boolean boolean0 = document0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document("&4|&wfV9,Qu");
      Element element0 = document0.toggleClass("&4|&wfV9,Qu");
      assertEquals("&4|&wfV9,Qu", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.toggleClass("");
      assertEquals("#root", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("kK&{!>^_YF)rSA.");
      String string0 = document0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("7d");
      Element element0 = document0.prependText("7d");
      element0.appendElement("7d");
      String string0 = document0.toString();
      assertEquals("7d<7d>\n</7d>", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("3yF)|1");
      document0.prependElement("3yF)|1");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "3yF)|1");
      document0.outerHtml(stringBuilder0);
      assertEquals("3yF)|1<#root>\n<3yf)|1>\n</3yf)|1>\n</#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("kK&{!>^_YpF)rSA.");
      boolean boolean0 = document0.equals("kK&{!>^_YpF)rSA.");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("yM$`?oHJW5;mT");
      document0.hashCode();
  }
}