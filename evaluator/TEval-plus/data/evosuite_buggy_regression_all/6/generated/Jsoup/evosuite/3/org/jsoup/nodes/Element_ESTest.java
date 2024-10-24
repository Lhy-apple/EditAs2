/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:31:11 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Comment;
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
      Document document0 = new Document("'");
      document0.appendElement("'");
      String string0 = document0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("l");
      Element element0 = document0.val("l");
      assertEquals("#document", element0.nodeName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("A 26VE5");
      Elements elements0 = document0.getElementsByClass("A 26VE5");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("^~=:`(|D$");
      // Undeclared exception!
      try { 
        document0.child(0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("'");
      Elements elements0 = document0.getElementsByAttributeValueContaining("'", "'");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("vNyD[MX5:");
      Element element0 = document0.createElement("vNyD[MX5:");
      Element element1 = element0.text("vNyD[MX5:");
      assertEquals("vnyd[mx5:", element1.tagName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("Y94");
      Elements elements0 = document0.getElementsByAttributeValueNot("Y94", "b");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("l");
      Elements elements0 = document0.getElementsByAttribute("l");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.addClass("vNyD[MX5:");
      boolean boolean0 = element0.hasClass("vNyD[MX5:");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("Kappa");
      // Undeclared exception!
      try { 
        document0.html("Kappa");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document(">$s9s8i4");
      Elements elements0 = document0.getElementsByAttributeValue(">$s9s8i4", ">$s9s8i4");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("");
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
  public void test12()  throws Throwable  {
      Document document0 = new Document("k");
      Elements elements0 = document0.getElementsByAttributeValueStarting("k", "k");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("174");
      Element element0 = document0.createElement("174");
      String string0 = element0.nodeName();
      assertEquals("174", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("Kappa");
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

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document(">$s9s8i4");
      Elements elements0 = document0.getAllElements();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getElementsByIndexLessThan(2181);
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("Kappa");
      Elements elements0 = document0.parents();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("alpha");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-762));
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getElementsByIndexEquals((-2344));
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("alpha");
      Element element0 = document0.getElementById("alpha");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("Z<");
      Element element0 = document0.appendElement("Z<");
      element0.prependChild(document0);
      Elements elements0 = document0.parents();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document(">ss9s8i4");
      Comment comment0 = new Comment(">ss9s8i4", "mH/D>");
      Element element0 = document0.prependChild(comment0);
      Elements elements0 = element0.getElementsByAttributeValueEnding("mH/D>", "s+");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("Kappa");
      document0.prependChild(document0);
      Element element0 = document0.nextElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("Kappa");
      document0.prependElement("Kappa");
      document0.prependChild(document0);
      Element element0 = document0.nextElementSibling();
      assertEquals("Kappa", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("l");
      Element element0 = document0.appendElement("l");
      Element element1 = element0.previousElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("Zq<");
      Element element0 = document0.appendElement("Zq<");
      document0.prependChild(document0);
      Element element1 = element0.previousElementSibling();
      assertSame(element1, document0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("Kappa");
      Element element0 = document0.prependChild(document0);
      Element element1 = element0.firstElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("Kappa");
      Element element0 = document0.prependChild(document0);
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("Kappa");
      document0.prependChild(document0);
      Element element0 = document0.lastElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("KaFppa");
      Document document1 = new Document("KaFppa");
      document0.addChild(document1);
      document0.prependChild(document0);
      Element element0 = document0.lastElementSibling();
      assertNotNull(element0);
      assertSame(element0, document1);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("'");
      Element element0 = document0.appendText("'");
      element0.prependText(" ");
      String string0 = element0.text();
      assertEquals("'", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document(">ss9s8i4");
      Comment comment0 = new Comment(">ss9s8i4", "mH/D>");
      document0.prependChild(comment0);
      String string0 = document0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("'");
      Element element0 = document0.appendText("'");
      document0.appendElement("'");
      String string0 = element0.text();
      assertEquals("'", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("l");
      Element element0 = document0.appendElement("l");
      boolean boolean0 = element0.preserveWhitespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.prependText("");
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("l");
      DataNode dataNode0 = new DataNode("l", "l");
      Element element0 = document0.prependChild(dataNode0);
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document(">$s9s8i4");
      Element element0 = document0.createElement(">$s9s8i4");
      element0.prependChild(document0);
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("Z<");
      Element element0 = document0.appendText("Z<");
      Element element1 = element0.appendElement("Z<");
      Element element2 = element1.prependChild(document0);
      boolean boolean0 = element2.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("'");
      document0.appendText("'");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document(">$s9s8i4");
      DataNode dataNode0 = new DataNode("T*Q}${ES$4.", "T*Q}${ES$4.");
      document0.prependChild(dataNode0);
      String string0 = document0.data();
      assertEquals("T*Q}${ES$4.", string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("'");
      document0.appendElement("'");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("Kappa");
      Element element0 = document0.removeClass("Kappa");
      String string0 = element0.className();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("7f");
      Element element0 = document0.toggleClass("7f");
      assertFalse(element0.isBlock());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.toggleClass("");
      assertSame(element0, document0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document(">$s9s8i4");
      String string0 = document0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("LINK");
      document0.appendElement("LINK");
      String string0 = document0.toString();
      assertEquals("<link />", string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("Zq<");
      Element element0 = document0.appendText("Zq<");
      element0.appendElement("Zq<");
      String string0 = document0.toString();
      assertEquals("Zq&lt;<zq<>\n</zq<>", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("l");
      Tag tag0 = Tag.valueOf("l");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, " wG){\"9YWF", attributes0);
      element0.prependChild(document0);
      String string0 = element0.toString();
      assertEquals("<l>\n<#root>\n</#root>\n</l>", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("");
      document0.appendElement("textarea");
      String string0 = document0.toString();
      assertEquals("<textarea></textarea>", string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("k");
      boolean boolean0 = document0.equals("k");
      assertFalse(boolean0);
  }
}
