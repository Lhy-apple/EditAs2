/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:39:34 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.dom;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.dom.DOMAttributeIterator;
import org.apache.html.dom.HTMLBRElementImpl;
import org.apache.html.dom.HTMLDocumentImpl;
import org.apache.html.dom.HTMLFrameSetElementImpl;
import org.apache.wml.dom.WMLDocumentImpl;
import org.apache.wml.dom.WMLGoElementImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;
import org.w3c.dom.Attr;
import org.w3c.dom.DocumentType;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DOMAttributeIterator_ESTest extends DOMAttributeIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      QName qName0 = new QName("Z.$V78.XBML3OO7,:", "Z.$V78.XBML3OO7,:");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl((DocumentType) null);
      WMLGoElementImpl wMLGoElementImpl0 = new WMLGoElementImpl(wMLDocumentImpl0, "7Uw<[B#~nk>L");
      Locale locale0 = Locale.ROOT;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, wMLGoElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      int int0 = dOMAttributeIterator0.getPosition();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      QName qName0 = new QName("Z.V78.XBML3OO,:");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLBRElementImpl hTMLBRElementImpl0 = new HTMLBRElementImpl(hTMLDocumentImpl0, "Z.V78.XBML3OO,:");
      Locale locale0 = Locale.ENGLISH;
      Attr attr0 = hTMLDocumentImpl0.createAttributeNS("Z.V78.XBML3OO,:", "Z.V78.XBML3OO,:", "Z.V78.XBML3OO,:");
      hTMLBRElementImpl0.setAttributeNode(attr0);
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLBRElementImpl0, locale0);
      QName qName1 = new QName("Z.V78.XBML3OO,:", "*");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName1);
      NodePointer nodePointer1 = dOMAttributeIterator0.getNodePointer();
      assertNotNull(nodePointer1);
      
      DOMAttributeIterator dOMAttributeIterator1 = new DOMAttributeIterator(nodePointer1, qName1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      QName qName0 = new QName("cite", "cite");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLBRElementImpl hTMLBRElementImpl0 = new HTMLBRElementImpl(hTMLDocumentImpl0, "cite");
      Locale locale0 = Locale.ENGLISH;
      Attr attr0 = hTMLDocumentImpl0.createAttributeNS("cite", "cite", "cite");
      hTMLBRElementImpl0.setAttributeNode(attr0);
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLBRElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      QName qName0 = new QName(".Z.$V;78XBL3O<O|,:", ".Z.$V;78XBL3O<O|,:");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLBRElementImpl hTMLBRElementImpl0 = new HTMLBRElementImpl(hTMLDocumentImpl0, ".Z.$V;78XBL3O<O|,:");
      Locale locale0 = Locale.ITALIAN;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLBRElementImpl0, locale0);
      QName qName1 = new QName("*");
      Attr attr0 = hTMLDocumentImpl0.createAttributeNS("t-*FCMP+/f)&qGcv4#", " to ", "xmlns");
      hTMLBRElementImpl0.setAttributeNodeNS(attr0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      QName qName0 = new QName("Z.$V78.XBL3OO;,:", "Z.$V78.XBL3OO;,:");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLBRElementImpl hTMLBRElementImpl0 = new HTMLBRElementImpl(hTMLDocumentImpl0, "Z.$V78.XBL3OO;,:");
      Locale locale0 = Locale.ITALIAN;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLBRElementImpl0, locale0);
      QName qName1 = new QName("xmlns", "<<unknown namespace>>");
      Attr attr0 = hTMLDocumentImpl0.createAttributeNS("Z.$V78.XBL3OO;,:", "<<unknown namespace>>", "<<unknown namespace>>");
      hTMLBRElementImpl0.setAttributeNodeNS(attr0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      QName qName0 = new QName(".V783XBML3O,:");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLBRElementImpl hTMLBRElementImpl0 = new HTMLBRElementImpl(hTMLDocumentImpl0, ".V783XBML3O,:");
      Locale locale0 = Locale.UK;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLBRElementImpl0, locale0);
      QName qName1 = new QName("xmlns", "<<unknown namespace>>");
      hTMLBRElementImpl0.setTitle("http://www.w3.org/XML/1998/namespace");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      QName qName0 = new QName("*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLBRElementImpl hTMLBRElementImpl0 = new HTMLBRElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.UK;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLBRElementImpl0, locale0);
      hTMLBRElementImpl0.setTitle("http://www.w3.org/XML/1998/namespace");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      QName qName0 = new QName("Z.VN8XML3,:");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLBRElementImpl hTMLBRElementImpl0 = new HTMLBRElementImpl(hTMLDocumentImpl0, "Z.VN8XML3,:");
      Locale locale0 = Locale.ENGLISH;
      Attr attr0 = hTMLDocumentImpl0.createAttributeNS("Z.VN8XML3,:", "Z.VN8XML3,:", "Z.VN8XML3,:");
      hTMLBRElementImpl0.setAttributeNode(attr0);
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLBRElementImpl0, locale0);
      QName qName1 = new QName("*");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      QName qName0 = new QName("^LZR");
      HTMLFrameSetElementImpl hTMLFrameSetElementImpl0 = new HTMLFrameSetElementImpl((HTMLDocumentImpl) null, "^LZR");
      Locale locale0 = Locale.CANADA;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLFrameSetElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = null;
      try {
        dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      QName qName0 = new QName("Z.$V78.XBML3OO7,:");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl((DocumentType) null);
      WMLGoElementImpl wMLGoElementImpl0 = new WMLGoElementImpl(wMLDocumentImpl0, "Z.$V78.XBML3OO7,:");
      Locale locale0 = Locale.ROOT;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, wMLGoElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      dOMAttributeIterator0.setPosition((-2021492923));
      // Undeclared exception!
      try { 
        dOMAttributeIterator0.getNodePointer();
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      QName qName0 = new QName("Z.$V78.XBML3OO7,:", "Z.$V78.XBML3OO7,:");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl((DocumentType) null);
      WMLGoElementImpl wMLGoElementImpl0 = new WMLGoElementImpl(wMLDocumentImpl0, "Z.$V78.XBML3OO7,:");
      Locale locale0 = Locale.ENGLISH;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, wMLGoElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      dOMAttributeIterator0.getNodePointer();
      // Undeclared exception!
      try { 
        dOMAttributeIterator0.getNodePointer();
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }
}
