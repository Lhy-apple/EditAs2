/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:07:57 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.regex.Pattern;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Entities;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("A");
      Document document1 = (Document)document0.appendText("A");
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("id");
      Elements elements0 = document0.getElementsMatchingText("id");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      Element element0 = document0.prependElement("br");
      assertEquals("br", element0.tagName());
      
      Element element1 = document0.prependText("=vsT");
      assertEquals(0, element1.siblingIndex());
      
      String string0 = element1.text();
      assertEquals("=vsT", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("fgo0Z=*yoc!");
      // Undeclared exception!
      try { 
        document0.child(64);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 64, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("fg<g8ZR=|yoc!b");
      Element element0 = document0.prependElement("br");
      element0.text("fg<g8ZR=|yoc!b");
      String string0 = document0.toString();
      assertEquals("<br>fg&lt;g8ZR=|yoc!b</br>", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("s");
      document0.tagName("s");
      document0.prependElement("s");
      Element element0 = document0.prependText("s");
      String string0 = element0.toString();
      assertEquals("s<s></s>", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      Map<String, String> map0 = document0.dataset();
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("r");
      Elements elements0 = document0.getElementsContainingOwnText("r");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("id");
      // Undeclared exception!
      try { 
        document0.html("id");
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
      Document document0 = new Document("e");
      Elements elements0 = document0.getElementsByAttributeValue("e", "e");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      Elements elements0 = document0.getElementsByAttributeValueStarting("br", "fgg8Zd=|yoc!b");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("}yo5h+G'N4<fSD*'g,");
      // Undeclared exception!
      try { 
        document0.select("}yo5h+G'N4<fSD*'g,");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query '}yo5h+G'N4<fSD*'g,': unexpected token at '}yo5h+G'N4<fSD*'g,'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("ih");
      Element element0 = document0.prepend("ih");
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("s");
      // Undeclared exception!
      try { 
        document0.after("s");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("nZF");
      Elements elements0 = document0.getElementsByIndexLessThan((-14));
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document("7P1r&WSIz:i^N");
      Elements elements0 = document0.getElementsByAttributeStarting("7P1r&WSIz:i^N");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("}yo5h+)G'N4<S*'g,");
      Elements elements0 = document0.getElementsByIndexEquals(2147483645);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
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
  public void test18()  throws Throwable  {
      Document document0 = new Document("A");
      Element element0 = document0.val("A");
      assertEquals("A", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("f!C!>]%O$OR#ieH00h");
      Elements elements0 = document0.getElementsByAttributeValueMatching("f!C!>]%O$OR#ieH00h", "f!C!>]%O$OR#ieH00h");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("e");
      Elements elements0 = document0.getElementsByAttributeValueEnding("e", "e");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("3Y6x[hqL<QhL+");
      document0.getElementsByClass("3Y6x[hqL<QhL+");
      Element element0 = document0.addClass("3Y6x[hqL<QhL+");
      assertFalse(element0.isBlock());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("[B}yo5h+G'N4<fS*g,");
      Elements elements0 = document0.getElementsByAttributeValueContaining("[B}yo5h+G'N4<fS*g,", "[B}yo5h+G'N4<fS*g,");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("I'WdQ]f[c");
      Elements elements0 = document0.getElementsByAttributeValueNot("I'WdQ]f[c", "I'WdQ]f[c");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document(",7%-5)3BU?$j#I8|4f");
      Elements elements0 = document0.getElementsByAttribute(",7%-5)3BU?$j#I8|4f");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("og.jsoup.parser.tmlTree{uilderSNate$24");
      // Undeclared exception!
      try { 
        document0.before("og.jsoup.parser.tmlTree{uilderSNate$24");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("VXduYXJ.;,q0%?jlO8,");
      Element element0 = document0.removeClass("VXduYXJ.;,q0%?jlO8,");
      assertEquals("#document", element0.nodeName());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("id");
      Elements elements0 = document0.getAllElements();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("fg<g8ZR=|yoc!b");
      Elements elements0 = document0.getElementsByTag("fg<g8ZR=|yoc!b");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("j8six");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1352));
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("fg<g8ZR=|yoc!b");
      Element element0 = document0.getElementById("&gt;t%U_&gt;h;/' \n<br>fg&lt;g8ZR=|yoc!b</br>");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("}yo5h+G'N4<fSD*'g,");
      Element element0 = document0.prependElement("}yo5h+G'N4<fSD*'g,");
      element0.appendChild(document0);
      // Undeclared exception!
      try { 
        element0.wrap("}yo5h+G'N4<fSD*'g,");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("}yo5h+G'N4<fSD*'g,");
      document0.appendChild(document0);
      List<TextNode> list0 = document0.textNodes();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("zM");
      Element element0 = document0.append("zM");
      List<TextNode> list0 = element0.textNodes();
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("vxduyxj.;,q0%?jlo8;,");
      Element element0 = document0.prependText("vxduyxj.;,q0%?jlo8;,");
      List<DataNode> list0 = element0.dataNodes();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("7P1r&WSIz:i^N");
      DataNode dataNode0 = new DataNode("7P1r&WSIz:i^N", "7P1r&WSIz:i^N");
      document0.appendChild(dataNode0);
      List<DataNode> list0 = document0.dataNodes();
      assertTrue(list0.contains(dataNode0));
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("Cookie");
      LinkedHashSet<Comment> linkedHashSet0 = new LinkedHashSet<Comment>();
      // Undeclared exception!
      try { 
        document0.insertChildren(43, linkedHashSet0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document("e");
      LinkedHashSet<Document> linkedHashSet0 = new LinkedHashSet<Document>();
      Element element0 = document0.insertChildren((-1), linkedHashSet0);
      assertSame(element0, document0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("Rl}()R^&4SB:b?Sh");
      LinkedList<FormElement> linkedList0 = new LinkedList<FormElement>();
      // Undeclared exception!
      try { 
        document0.insertChildren((-1618280579), linkedList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("f6gZZd=|yoc!b");
      Elements elements0 = document0.siblingElements();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("f6gZZd=|yoc!b");
      Element element0 = document0.prependElement("f6gZZd=|yoc!b");
      Element element1 = element0.before((Node) document0);
      Elements elements0 = element1.siblingElements();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("org.jsoup.nodes.Element");
      Element element0 = document0.prependElement("tfod`Dot");
      Element element1 = element0.nextElementSibling();
      assertEquals("tfod`dot", element0.nodeName());
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("f!C!>]%O$OR#ieH00h");
      Element element0 = document0.nextElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("org.jsoup.nodes.Element");
      Element element0 = document0.prependElement("tfod`Dot");
      element0.after((Node) document0);
      Element element1 = element0.nextElementSibling();
      assertEquals("tfod`dot", element0.nodeName());
      assertNotNull(element1);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("%");
      Element element0 = document0.prependElement("%");
      Element element1 = element0.previousElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document("7P1r&WSIz:i^N");
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("<]\",gP:3FY|8){+svr");
      Document document1 = document0.clone();
      Node[] nodeArray0 = new Node[8];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document1;
      nodeArray0[2] = (Node) document1;
      nodeArray0[3] = (Node) document1;
      nodeArray0[4] = (Node) document1;
      nodeArray0[5] = (Node) document1;
      nodeArray0[6] = (Node) document0;
      nodeArray0[7] = (Node) document1;
      document0.addChildren(nodeArray0);
      document1.previousElementSibling();
      assertEquals(1, document1.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("o5h+G'N4<fSD*'g,");
      Element element0 = document0.prependElement("o5h+G'N4<fSD*'g,");
      Element element1 = element0.firstElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yc!b");
      Element element0 = document0.createElement("fgg8Zd=|yc!b");
      Node[] nodeArray0 = new Node[5];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document0;
      Comment comment0 = new Comment("fgg8Zd=|yc!b", "fgg8Zd=|yc!b");
      nodeArray0[2] = (Node) comment0;
      nodeArray0[3] = (Node) document0;
      nodeArray0[4] = (Node) element0;
      element0.addChildren(nodeArray0);
      Element element1 = element0.firstElementSibling();
      assertSame(document0, element1);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("*?(;OU!$y-_!0n/)/Ny");
      Element element0 = document0.prependElement("*?(;OU!$y-_!0n/)/Ny");
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      Element element0 = document0.prependElement("fgg8Zd=|yoc!b");
      Element element1 = element0.lastElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("f6gZZd=|yoc!b");
      Element element0 = document0.prependElement("f6gZZd=|yoc!b");
      element0.before((Node) document0);
      Element element1 = document0.lastElementSibling();
      assertEquals("f6gZZd=|yoc!b", element1.baseUri());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("7P1r&SIz:i^N");
      document0.parentNode = (Node) document0;
      // Undeclared exception!
      try { 
        document0.previousElementSibling();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      DataNode dataNode0 = DataNode.createFromEncoded("br", "br");
      document0.appendChild(dataNode0);
      String string0 = document0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("s");
      document0.prependElement("s");
      document0.prependText("Pattern syntax error: ");
      Elements elements0 = document0.getElementsContainingText("n9o@ynnlc\"");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Document document0 = new Document("e");
      document0.prependElement("e");
      document0.prependText("e");
      Elements elements0 = document0.getElementsContainingText("e");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Document document0 = new Document("fg<g8ZR=|yoc!b");
      Element element0 = document0.prependElement("br");
      element0.text("fg<g8ZR=|yoc!b");
      Pattern pattern0 = Pattern.compile("&gt;t%U_&gt;h;/' \n<br>fg&lt;g8ZR=|yoc!b</br>", (-1268));
      Elements elements0 = document0.getElementsMatchingOwnText(pattern0);
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = new Document("I'Wd]f[c");
      DocumentType documentType0 = new DocumentType("I'Wd]f[c", "I'Wd]f[c", "fB#1&`(SG%", "I'Wd]f[c");
      Element element0 = document0.appendChild(documentType0);
      Elements elements0 = element0.getElementsMatchingOwnText("OS3");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = new Document("o5h+G'N4<fSD*'g,");
      Element element0 = document0.prependElement("textarea");
      Element element1 = element0.text("textarea");
      element1.getElementsMatchingOwnText("o5h+G'N4<fSD*'g,");
      assertEquals("textarea", element1.tagName());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document(":");
      document0.prependElement(":");
      Elements elements0 = document0.getElementsMatchingOwnText(":");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Document document0 = new Document("fg<g8ZR=|yoc!b");
      document0.prependElement("br");
      document0.prependText(">t%U_>h;/' ");
      Pattern pattern0 = Pattern.compile("&gt;t%U_&gt;h;/' \n<br>fg&lt;g8ZR=|yoc!b</br>", (-1268));
      Elements elements0 = document0.getElementsMatchingOwnText(pattern0);
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      boolean boolean0 = Element.preserveWhitespace((Node) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      TextNode textNode0 = new TextNode("ge5U;^#5Q\"l}a{b", "ge5U;^#5Q\"l}a{b");
      boolean boolean0 = Element.preserveWhitespace(textNode0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Document document0 = new Document("s");
      Element element0 = document0.prependElement("textarea");
      element0.prependChild(document0);
      boolean boolean0 = Element.preserveWhitespace(document0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Document document0 = new Document("I'Wd]f[c");
      DocumentType documentType0 = new DocumentType("I'Wd]f[c", "I'Wd]f[c", "fB#1&`(SG%", "I'Wd]f[c");
      document0.appendChild(documentType0);
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.prependElement("textarea");
      element0.text("");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Document document0 = new Document("org.jsoup.nodes.Element");
      Element element0 = document0.prependElement("textarea");
      element0.text("org.jsoup.nodes.Element");
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Document document0 = new Document("'");
      DataNode dataNode0 = DataNode.createFromEncoded("'", "");
      document0.appendChild(dataNode0);
      String string0 = document0.data();
      assertEquals("'", string0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      Element element0 = document0.prependElement("fgg8Zd=|yoc!b");
      element0.text("fgg8Zd=|yoc!b");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Document document0 = new Document("");
      boolean boolean0 = document0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Document document0 = new Document("nZF");
      Element element0 = document0.toggleClass("nZF");
      assertEquals(0, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.toggleClass("");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Document document0 = new Document("V>Fk^~b'FlYb0f.r");
      String string0 = document0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Document document0 = new Document("Malformed URL: ");
      document0.tagName("textarea");
      document0.val();
      assertEquals("textarea", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Document document0 = new Document("ofg.j&oup.nodes.Element");
      Element element0 = document0.prependElement("textarea");
      element0.val(" />");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Document document0 = new Document("fg<g8ZR=|yoc!b");
      document0.prependElement("br");
      document0.prependText(">t%U_>h;/' ");
      String string0 = document0.toString();
      assertEquals("&gt;t%U_&gt;h;/' \n<br />", string0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Document document0 = new Document(":matches(regex) query must not be empty");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      Document document1 = document0.outputSettings(document_OutputSettings1);
      document0.appendChild(document1);
      // Undeclared exception!
      document1.toString();
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Document document0 = new Document("}yo5h+G'N4<fSD*'g,");
      document0.appendChild(document0);
      // Undeclared exception!
      document0.toString();
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Document document0 = new Document("value");
      Element element0 = document0.prependElement("textarea");
      StringBuilder stringBuilder0 = new StringBuilder();
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Element element1 = element0.clone();
      element0.outerHtmlTail(stringBuilder0, 2271, document_OutputSettings0);
      element1.outerHtmlHead(stringBuilder0, (-1968742937), document_OutputSettings0);
      assertEquals("</textarea><textarea>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Document document0 = new Document("value");
      Element element0 = document0.prependElement("textarea");
      StringBuilder stringBuilder0 = new StringBuilder();
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Element element1 = element0.clone();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      element0.outerHtmlTail(stringBuilder0, 2271, document_OutputSettings0);
      // Undeclared exception!
      try { 
        element1.outerHtmlHead(stringBuilder0, (-1968742937), document_OutputSettings1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // width must be > 0
         //
         verifyException("org.jsoup.helper.StringUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Document document0 = new Document("id");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      // Undeclared exception!
      try { 
        document0.outerHtmlTail((StringBuilder) null, (-2021161078), document_OutputSettings1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Element", e);
      }
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      Element element0 = document0.prependElement("fgg8Zd=|yoc!b");
      element0.text("fgg8Zd=|yoc!b");
      String string0 = element0.toString();
      assertEquals("<fgg8zd=|yoc!b>\n fgg8Zd=|yoc!b\n</fgg8zd=|yoc!b>", string0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      Element element0 = document0.prependElement("br");
      element0.text("fgg8Zd=|yoc!b");
      StringBuilder stringBuilder0 = new StringBuilder();
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      element0.outerHtmlTail(stringBuilder0, 991, document_OutputSettings1);
      assertEquals("</br>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      Element element0 = document0.prependElement("br");
      Element element1 = element0.text("fgg8Zd=|yoc!b");
      StringBuilder stringBuilder0 = new StringBuilder();
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      DataNode dataNode0 = DataNode.createFromEncoded("br", "br");
      element1.appendChild(dataNode0);
      element0.outerHtmlTail(stringBuilder0, 991, document_OutputSettings1);
      assertEquals("\n                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               </br>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.prependElement("textarea");
      StringBuilder stringBuilder0 = new StringBuilder();
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      element0.appendChild(document0);
      element0.outerHtmlTail(stringBuilder0, 1363, document_OutputSettings1);
      assertEquals(Entities.EscapeMode.base, document_OutputSettings1.escapeMode());
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      Document document0 = new Document("fgg8Zd=|yoc!b");
      document0.hashCode();
  }
}